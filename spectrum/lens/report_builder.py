"""
Report Builder for Compliance Artifacts

Generates DOCX/PDF/HTML reports from audit data using Jinja2 templates and python-docx.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from loguru import logger

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not installed. DOCX generation will not be available.")


class ReportBuilder:
    """
    Builds compliance reports from audit data.

    Supports multiple output formats:
    - DOCX: Professional Word documents with formatting
    - HTML: Web-based reports using Jinja2 templates
    - PDF: Planned (currently generates HTML)
    """

    def __init__(self, template_dir: str = None):
        """
        Initialize the report builder.

        Args:
            template_dir: Directory containing Jinja2 templates.
                        Defaults to 'templates' directory in the spectrum package.
        """
        if template_dir is None:
            # Default to templates directory in the package
            package_dir = Path(__file__).parent.parent
            template_dir = str(package_dir / "templates")

        self.template_dir = template_dir

        # Ensure template directory exists
        if not os.path.exists(self.template_dir):
            logger.warning(f"Template directory does not exist: {self.template_dir}")
            os.makedirs(self.template_dir, exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )

    def generate_docx(
        self,
        data: Dict[str, Any],
        output_path: str,
        template_type: str = "tier1_forensic_audit"
    ) -> str:
        """
        Generate professional DOCX report from audit data.

        Args:
            data: Dictionary of audit data
            output_path: Path to save the DOCX file
            template_type: Type of report template to use

        Returns:
            Path to the generated DOCX file

        Raises:
            ImportError: If python-docx is not installed
        """
        if not DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is required for DOCX generation. "
                "Install it with: pip install python-docx"
            )

        # Route to appropriate template generator
        if template_type == "tier1_forensic_audit":
            return self._generate_tier1_audit_docx(data, output_path)
        elif template_type == "eu_ai_act_annex_iv":
            return self._generate_annex_iv_docx(data, output_path)
        elif template_type == "cfpb_model_validation":
            return self._generate_cfpb_docx(data, output_path)
        else:
            # Fallback to generic template
            return self._generate_generic_docx(data, output_path)

    def _generate_tier1_audit_docx(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Generate Tier 1 Forensic Audit report in DOCX format.

        This is the standard comprehensive audit report.
        """
        doc = Document()

        # Configure default styles
        self._configure_docx_styles(doc)

        # === COVER PAGE ===
        self._add_cover_page(doc, data)

        # === EXECUTIVE SUMMARY ===
        doc.add_page_break()
        self._add_executive_summary(doc, data)

        # === SCOPE & METHODOLOGY ===
        doc.add_page_break()
        self._add_scope_methodology(doc, data)

        # === ROBUSTNESS ASSESSMENT ===
        doc.add_page_break()
        self._add_robustness_assessment(doc, data)

        # === EXPLAINABILITY ANALYSIS ===
        doc.add_page_break()
        self._add_explainability_analysis(doc, data)

        # === DRIFT MONITORING ===
        doc.add_page_break()
        self._add_drift_monitoring(doc, data)

        # === UNCERTAINTY QUANTIFICATION ===
        doc.add_page_break()
        self._add_uncertainty_quantification(doc, data)

        # === REGULATORY MAPPING ===
        doc.add_page_break()
        self._add_regulatory_mapping(doc, data)

        # === RECOMMENDATIONS ===
        doc.add_page_break()
        self._add_recommendations(doc, data)

        # === TECHNICAL APPENDIX ===
        doc.add_page_break()
        self._add_technical_appendix(doc, data)

        # Save document
        doc.save(output_path)
        logger.info(f"DOCX report generated: {output_path}")

        return output_path

    def _configure_docx_styles(self, doc: Document):
        """Configure document-wide styles."""
        # Configure Normal style
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)

        # Configure Heading styles
        for i in range(1, 4):
            heading_style = doc.styles[f'Heading {i}']
            heading_font = heading_style.font
            heading_font.name = 'Calibri'
            heading_font.bold = True
            heading_font.color.rgb = RGBColor(0, 51, 102)  # Dark blue

    def _add_cover_page(self, doc: Document, data: Dict[str, Any]):
        """Add professional cover page."""
        # Title
        title = doc.add_heading('Model Governance Compliance Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add spacing
        doc.add_paragraph()
        doc.add_paragraph()

        # Subtitle
        subtitle = doc.add_paragraph('Tier 1 Forensic Algorithmic Audit')
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_format = subtitle.runs[0]
        subtitle_format.font.size = Pt(16)
        subtitle_format.font.color.rgb = RGBColor(128, 128, 128)

        # Add more spacing
        doc.add_paragraph()
        doc.add_paragraph()

        # Model information
        model_info = doc.add_paragraph()
        model_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
        model_info.add_run(f"Model: {data.get('model_name', 'N/A')}\n").bold = True
        model_info.add_run(f"Type: {data.get('model_type', 'N/A')}\n")
        model_info.add_run(f"Risk Level: {data.get('risk_level', 'N/A')}\n")

        # Add spacing
        doc.add_paragraph()
        doc.add_paragraph()

        # Client and date information
        client_info = doc.add_paragraph()
        client_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
        client_info.add_run(f"Client: {data.get('client', 'N/A')}\n")
        client_info.add_run(f"Audit Date: {data.get('timestamp', datetime.now().strftime('%Y-%m-%d'))}\n")

        # Regulatory context
        if 'regulatory_context' in data:
            contexts = data['regulatory_context']
            if isinstance(contexts, list):
                contexts = ', '.join(contexts)
            doc.add_paragraph()
            reg_para = doc.add_paragraph()
            reg_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            reg_para.add_run(f"Regulatory Context: {contexts}")

    def _add_executive_summary(self, doc: Document, data: Dict[str, Any]):
        """Add executive summary section."""
        doc.add_heading('Executive Summary', 1)

        # Overall compliance status
        doc.add_heading('Compliance Status', 2)
        status_para = doc.add_paragraph()

        # Extract adversarial metrics
        adv_metrics = data.get('adversarial_metrics', {})
        attack_success_rate = adv_metrics.get('attack_success_rate')

        # Determine overall status
        coverage_pass = data.get('empirical_coverage', 0) >= data.get('confidence_required', 0.95)
        robustness_pass = (attack_success_rate is not None) and (attack_success_rate < 0.30)  # Lower ASR = more robust
        drift_pass = not data.get('data_drift_alert', False)

        if coverage_pass and robustness_pass and drift_pass:
            status_run = status_para.add_run('✓ COMPLIANT')
            status_run.font.color.rgb = RGBColor(0, 128, 0)  # Green
            status_run.bold = True
            doc.add_paragraph('The model meets all governance requirements and is suitable for deployment.')
        else:
            status_run = status_para.add_run('⚠ REQUIRES ATTENTION')
            status_run.font.color.rgb = RGBColor(255, 140, 0)  # Orange
            status_run.bold = True
            doc.add_paragraph('The model has compliance gaps that require remediation before deployment.')

        # Key findings
        doc.add_heading('Key Findings', 2)

        findings = []

        # Robustness finding (using attack success rate)
        if attack_success_rate is None:
            findings.append(('ℹ', 'Robustness', 'No adversarial testing performed (optional red team dependencies not installed)'))
        elif attack_success_rate <= 0.30:
            findings.append(('✓', 'Robustness', f'Model demonstrates strong robustness (Attack Success Rate: {attack_success_rate:.1%})'))
        elif attack_success_rate <= 0.60:
            findings.append(('⚠', 'Robustness', f'Model shows moderate vulnerability (Attack Success Rate: {attack_success_rate:.1%})'))
        else:
            findings.append(('✗', 'Robustness', f'Model is highly vulnerable (Attack Success Rate: {attack_success_rate:.1%})'))

        # Coverage finding
        coverage = data.get('empirical_coverage', 0)
        required = data.get('confidence_required', 0.95)
        if coverage >= required:
            findings.append(('✓', 'Coverage', f'Uncertainty quantification meets requirements ({coverage:.1%} >= {required:.1%})'))
        else:
            findings.append(('✗', 'Coverage', f'Coverage below requirement ({coverage:.1%} < {required:.1%})'))

        # Drift finding
        if data.get('data_drift_alert', False):
            findings.append(('⚠', 'Drift', 'Significant data drift detected - immediate action required'))
        else:
            findings.append(('✓', 'Drift', data.get('data_drift_status', 'No significant drift detected')))

        # Add findings to document
        for symbol, category, description in findings:
            p = doc.add_paragraph(style='List Bullet')
            run = p.add_run(f'{symbol} ')
            run.bold = True
            p.add_run(f'{category}: {description}')

    def _add_scope_methodology(self, doc: Document, data: Dict[str, Any]):
        """Add scope and methodology section."""
        doc.add_heading('Scope & Methodology', 1)

        doc.add_heading('Audit Scope', 2)
        doc.add_paragraph(
            f"This audit assesses the {data.get('model_name', 'model')} "
            f"({data.get('model_type', 'N/A')}) for compliance with "
            f"{data.get('regulatory_context', 'applicable regulations')}."
        )

        doc.add_heading('Methodology', 2)
        doc.add_paragraph('The audit employed the following methodologies:')

        # Extract attack type from adversarial metrics
        adv_metrics = data.get('adversarial_metrics', {})
        attack_type = adv_metrics.get('attack_type', 'HopSkipJump')

        methodologies = [
            ('Adversarial Testing', f"Attack method: {attack_type}"),
            ('Uncertainty Quantification', 'Conformal prediction (MAPIE) with coverage guarantees'),
            ('Explainability Analysis', 'SHAP-based feature attribution'),
            ('Drift Monitoring', 'Population Stability Index (PSI) analysis'),
        ]

        for title, description in methodologies:
            p = doc.add_paragraph(style='List Bullet')
            p.add_run(f'{title}: ').bold = True
            p.add_run(description)

        doc.add_heading('Sample Characteristics', 2)
        doc.add_paragraph(f"Test samples: {adv_metrics.get('samples_tested', 'N/A')}")
        doc.add_paragraph(f"Audit timestamp: {data.get('timestamp', 'N/A')}")

    def _add_robustness_assessment(self, doc: Document, data: Dict[str, Any]):
        """Add robustness assessment section."""
        doc.add_heading('Robustness Assessment', 1)

        doc.add_paragraph(
            'This section evaluates the model\'s resilience to adversarial perturbations, '
            'addressing EU AI Act Article 15 robustness requirements.'
        )

        # Extract adversarial metrics
        adv_metrics = data.get('adversarial_metrics', {})
        attack_success_rate = adv_metrics.get('attack_success_rate')
        attack_type = adv_metrics.get('attack_type', 'N/A')
        samples_tested = adv_metrics.get('samples_tested', 0)
        samples_successful = adv_metrics.get('samples_successful', 0)
        emp_rob_l2 = adv_metrics.get('empirical_robustness_l2', 0)
        emp_rob_linf = adv_metrics.get('empirical_robustness_linf', 0)

        if attack_success_rate is None:
            # No adversarial testing performed
            doc.add_heading('Adversarial Testing Status', 2)
            p = doc.add_paragraph('ℹ No adversarial testing performed')
            p.add_run('\n\nNote: Adversarial testing requires optional red team dependencies (ART library). ')
            p.add_run('Install with: pip install spectrum-governance[red_torch]')
        else:
            # Adversarial testing was performed
            doc.add_heading('Attack Success Rate Analysis', 2)

            doc.add_paragraph(f'Attack Success Rate: {attack_success_rate:.1%}')
            doc.add_paragraph(f'Successful Attacks: {samples_successful} / {samples_tested} samples')

            # Interpretation
            if attack_success_rate <= 0.30:
                interpretation = 'GREEN - Robust: Model demonstrates strong robustness. Acceptable for high-risk deployment.'
                color = RGBColor(0, 128, 0)
            elif attack_success_rate <= 0.60:
                interpretation = 'YELLOW - Moderate: Model shows vulnerability. Recommend hardening before deployment.'
                color = RGBColor(255, 140, 0)
            else:
                interpretation = 'RED - Vulnerable: Model is highly susceptible to adversarial manipulation.'
                color = RGBColor(255, 0, 0)

            p = doc.add_paragraph()
            run = p.add_run(f'Risk Level: {interpretation}')
            run.font.color.rgb = color
            run.bold = True

            doc.add_heading('Attack Details', 2)
            doc.add_paragraph(f"Attack Method: {attack_type}")
            doc.add_paragraph(f"Samples Tested: {samples_tested}")

            doc.add_heading('Perturbation Magnitudes', 2)
            doc.add_paragraph(
                'These metrics measure how much input needs to be modified to fool the model. '
                'Lower values indicate easier attacks (higher vulnerability).'
            )
            doc.add_paragraph(f"Empirical Robustness (L2): {emp_rob_l2:.4f}")
            doc.add_paragraph(f"Empirical Robustness (L∞): {emp_rob_linf:.4f}")

    def _add_explainability_analysis(self, doc: Document, data: Dict[str, Any]):
        """Add explainability analysis section."""
        doc.add_heading('Explainability Analysis', 1)

        doc.add_paragraph(
            'This section evaluates the model\'s interpretability and alignment with '
            'CFPB adverse action notice requirements.'
        )

        doc.add_heading('Feature Importance', 2)

        if 'sample_adverse_reasons' in data and data['sample_adverse_reasons']:
            doc.add_paragraph('Top contributing features:')
            for i, reason in enumerate(data['sample_adverse_reasons'][:5], 1):
                doc.add_paragraph(f'{i}. {reason}', style='List Number')
        else:
            doc.add_paragraph('Feature importance data not available.')

        doc.add_heading('Regulatory Compliance', 2)
        doc.add_paragraph(
            'The model provides SHAP-based explanations that can be mapped to '
            'adverse action reasons for CFPB compliance.'
        )

    def _add_drift_monitoring(self, doc: Document, data: Dict[str, Any]):
        """Add drift monitoring section."""
        doc.add_heading('Data Drift Monitoring', 1)

        doc.add_paragraph(
            'Population Stability Index (PSI) analysis comparing model inputs '
            'to training distribution.'
        )

        doc.add_heading('Drift Status', 2)

        drift_status = data.get('data_drift_status', 'Unknown')
        alert_required = data.get('data_drift_alert', False)

        p = doc.add_paragraph(f'Status: {drift_status}')
        if alert_required:
            p.runs[0].font.color.rgb = RGBColor(255, 0, 0)
        else:
            p.runs[0].font.color.rgb = RGBColor(0, 128, 0)

        doc.add_paragraph(f'Alert Required: {alert_required}')

        doc.add_heading('Interpretation', 2)
        doc.add_paragraph('PSI Thresholds:')
        doc.add_paragraph('• < 0.10: No significant shift', style='List Bullet')
        doc.add_paragraph('• 0.10-0.25: Moderate shift (monitor)', style='List Bullet')
        doc.add_paragraph('• > 0.25: Significant shift (immediate action)', style='List Bullet')

    def _add_uncertainty_quantification(self, doc: Document, data: Dict[str, Any]):
        """Add uncertainty quantification section."""
        doc.add_heading('Uncertainty Quantification', 1)

        doc.add_paragraph(
            'Conformal prediction provides finite-sample coverage guarantees '
            'for model predictions.'
        )

        doc.add_heading('Coverage Analysis', 2)

        required = data.get('confidence_required', 0.95)
        empirical = data.get('empirical_coverage', 0)

        doc.add_paragraph(f'Required Confidence: {required:.1%}')
        doc.add_paragraph(f'Empirical Coverage: {empirical:.1%}')

        if empirical >= required:
            p = doc.add_paragraph('✓ Coverage requirement MET')
            p.runs[0].font.color.rgb = RGBColor(0, 128, 0)
            p.runs[0].bold = True
        else:
            p = doc.add_paragraph('✗ Coverage requirement NOT MET')
            p.runs[0].font.color.rgb = RGBColor(255, 0, 0)
            p.runs[0].bold = True

        doc.add_heading('Risk Classification', 2)
        risk_level = data.get('risk_level', 'N/A')
        doc.add_paragraph(f'Model Risk Level: {risk_level}')

        if risk_level == 'HIGH':
            doc.add_paragraph('HIGH risk classification requires 95% confidence (alpha=0.05)')
        elif risk_level == 'MEDIUM':
            doc.add_paragraph('MEDIUM risk classification requires 90% confidence (alpha=0.10)')
        elif risk_level == 'LOW':
            doc.add_paragraph('LOW risk classification requires 80% confidence (alpha=0.20)')

    def _add_regulatory_mapping(self, doc: Document, data: Dict[str, Any]):
        """Add regulatory mapping section."""
        doc.add_heading('Regulatory Mapping', 1)

        doc.add_paragraph(
            'This section maps audit findings to specific regulatory requirements.'
        )

        regulatory_context = data.get('regulatory_context', [])
        if isinstance(regulatory_context, str):
            regulatory_context = [regulatory_context]

        if 'EU_AI_ACT' in regulatory_context or 'EU AI Act' in str(regulatory_context):
            doc.add_heading('EU AI Act Article 15', 2)
            adv_metrics = data.get('adversarial_metrics', {})
            attack_success_rate = adv_metrics.get('attack_success_rate')
            if attack_success_rate is not None and isinstance(attack_success_rate, (int, float)):
                asr_text = f"{attack_success_rate:.1%}"
                meets_threshold = "meets" if attack_success_rate < 0.30 else "does not meet"
            else:
                asr_text = "N/A"
                meets_threshold = "cannot be determined without"

            doc.add_paragraph(
                f"Article 15 requires 'appropriate level of accuracy, robustness and cybersecurity'. "
                f"Attack Success Rate of {asr_text} "
                f"{meets_threshold} "
                f"typical robustness thresholds (< 30%)."
            )

        if 'CFPB' in regulatory_context:
            doc.add_heading('CFPB Adverse Action Requirements', 2)
            doc.add_paragraph(
                'The model provides SHAP-based explanations that can be mapped to '
                'specific adverse action reasons required by ECOA and Regulation B.'
            )

        if 'EEOC' in regulatory_context or 'HB_3773' in regulatory_context:
            doc.add_heading('Employment Discrimination Compliance', 2)
            doc.add_paragraph(
                'Note: Disparate impact analysis (4/5ths rule) is not yet implemented. '
                'This is a critical gap for employment screening compliance.'
            )

    def _add_recommendations(self, doc: Document, data: Dict[str, Any]):
        """Add recommendations section."""
        doc.add_heading('Recommendations', 1)

        doc.add_paragraph(
            'Prioritized action items based on audit findings:'
        )

        # Generate recommendations based on findings
        recommendations = []

        # Extract adversarial metrics
        adv_metrics = data.get('adversarial_metrics', {})
        attack_success_rate = adv_metrics.get('attack_success_rate')

        if attack_success_rate is not None:
            if attack_success_rate > 0.60:
                recommendations.append((
                    'CRITICAL',
                    'Address Model Vulnerability',
                    f'Attack Success Rate of {attack_success_rate:.1%} indicates high vulnerability. '
                    'Implement adversarial training or input sanitization before deployment.'
                ))
            elif attack_success_rate > 0.30:
                recommendations.append((
                    'HIGH',
                    'Enhance Model Robustness',
                    f'Attack Success Rate of {attack_success_rate:.1%} suggests moderate vulnerability. '
                    'Consider adversarial training to improve robustness.'
                ))

        coverage = data.get('empirical_coverage', 0)
        required = data.get('confidence_required', 0.95)
        if coverage < required:
            recommendations.append((
                'CRITICAL',
                'Improve Uncertainty Calibration',
                f'Empirical coverage ({coverage:.1%}) falls short of requirement ({required:.1%}). '
                'Expand calibration dataset or adjust prediction sets.'
            ))

        if data.get('data_drift_alert', False):
            recommendations.append((
                'HIGH',
                'Address Data Drift',
                'Significant drift detected. Retrain model on recent data or implement drift adaptation.'
            ))

        if not recommendations:
            doc.add_paragraph('✓ No critical issues identified. Model meets governance requirements.')
        else:
            for priority, title, description in recommendations:
                p = doc.add_paragraph()
                p.add_run(f'[{priority}] {title}\n').bold = True
                p.add_run(description)
                doc.add_paragraph()  # Spacing

    def _add_technical_appendix(self, doc: Document, data: Dict[str, Any]):
        """Add technical appendix."""
        doc.add_heading('Technical Appendix', 1)

        doc.add_heading('Audit Metadata', 2)
        doc.add_paragraph(f"Lineage Run ID: {data.get('lineage_run_id', 'N/A')}")
        doc.add_paragraph(f"Audit Log Path: {data.get('audit_log_path', 'N/A')}")
        doc.add_paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        doc.add_heading('Methodology Details', 2)
        doc.add_paragraph(
            'Full methodology documentation and attack parameters are available '
            'in the RCIA audit logs.'
        )

        doc.add_heading('Limitations', 2)
        limitations = [
            'Adversarial testing limited to HopSkipJump attack (other attacks not yet implemented)',
            'Disparate impact analysis (fairness) not yet implemented',
            'LLM-specific security probes not yet implemented',
            'Feature constraints not enforced in adversarial testing',
        ]

        for limitation in limitations:
            doc.add_paragraph(f'• {limitation}', style='List Bullet')

    def _generate_annex_iv_docx(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Generate EU AI Act Annex IV Technical Documentation.

        Placeholder for future implementation.
        """
        doc = Document()
        doc.add_heading('EU AI Act Annex IV Technical Documentation', 0)
        doc.add_paragraph('This template is under development.')
        doc.add_paragraph(
            'Annex IV requires detailed technical documentation including design specifications, '
            'validation procedures, risk management measures, and change logs.'
        )
        doc.save(output_path)
        logger.info(f"Annex IV DOCX (placeholder) generated: {output_path}")
        return output_path

    def _generate_cfpb_docx(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Generate CFPB Model Validation Report.

        Placeholder for future implementation.
        """
        doc = Document()
        doc.add_heading('CFPB Model Validation Report', 0)
        doc.add_paragraph('This template is under development.')
        doc.add_paragraph(
            'CFPB model validation requires documentation of model development, validation, '
            'adverse action compliance, and ongoing monitoring.'
        )
        doc.save(output_path)
        logger.info(f"CFPB DOCX (placeholder) generated: {output_path}")
        return output_path

    def _generate_generic_docx(self, data: Dict[str, Any], output_path: str) -> str:
        """Generate generic DOCX report as fallback."""
        doc = Document()

        # Title
        doc.add_heading('Model Governance Audit Report', 0)

        # Basic information
        doc.add_heading('Model Information', 1)
        doc.add_paragraph(f"Model Name: {data.get('model_name', 'N/A')}")
        doc.add_paragraph(f"Model Type: {data.get('model_type', 'N/A')}")
        doc.add_paragraph(f"Risk Level: {data.get('risk_level', 'N/A')}")

        # Performance metrics
        doc.add_heading('Performance Metrics', 1)
        doc.add_paragraph(f"Confidence Required: {data.get('confidence_required', 'N/A')}")
        doc.add_paragraph(f"Empirical Coverage: {data.get('empirical_coverage', 'N/A')}")

        adv_metrics = data.get('adversarial_metrics', {})
        attack_success_rate = adv_metrics.get('attack_success_rate', 'N/A')
        doc.add_paragraph(f"Attack Success Rate: {attack_success_rate}")

        # Save
        doc.save(output_path)
        logger.info(f"Generic DOCX report generated: {output_path}")
        return output_path

    def generate_html(
        self,
        template_name: str,
        data: Dict[str, Any],
        output_path: str = None
    ) -> str:
        """
        Generate HTML report from template.

        Args:
            template_name: Name of the Jinja2 template file
            data: Dictionary of data to populate the template
            output_path: Optional path to save the HTML file

        Returns:
            Generated HTML string

        Raises:
            TemplateNotFound: If the template file doesn't exist
        """
        try:
            template = self.env.get_template(template_name)
            html_content = template.render(**data)

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML report generated: {output_path}")

            return html_content

        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            # Generate basic fallback HTML
            return self._generate_fallback_html(data, output_path)

    def generate_pdf(
        self,
        template_name: str,
        data: Dict[str, Any],
        output_path: str
    ):
        """
        Generate PDF report from template.

        Note: This currently generates HTML as a fallback.
        For PDF generation, you would need to install a library like:
        - weasyprint: pip install weasyprint
        - pdfkit: pip install pdfkit (requires wkhtmltopdf)

        Args:
            template_name: Name of the Jinja2 template file
            data: Dictionary of data to populate the template
            output_path: Path to save the PDF file
        """
        # Generate HTML first
        html_path = output_path.replace('.pdf', '.html')
        html_content = self.generate_html(template_name, data, html_path)

        # TODO: Convert HTML to PDF
        # For now, we just create an HTML file as a fallback
        logger.warning(
            f"PDF generation not yet implemented. "
            f"HTML report saved to: {html_path}"
        )

        # Placeholder for future PDF generation:
        # try:
        #     import weasyprint
        #     weasyprint.HTML(string=html_content).write_pdf(output_path)
        #     logger.info(f"PDF report generated: {output_path}")
        # except ImportError:
        #     logger.error("weasyprint not installed. Install with: pip install weasyprint")

    def _generate_fallback_html(self, data: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate a basic HTML report when template is missing.

        Args:
            data: Audit data dictionary
            output_path: Optional path to save the HTML

        Returns:
            Basic HTML string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Compliance Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; }}
        .section {{ margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Governance Compliance Report</h1>
        <p>Generated: {data.get('timestamp', 'N/A')}</p>
    </div>

    <div class="section">
        <h2>Model Information</h2>
        <table>
            <tr><th>Field</th><th>Value</th></tr>
            <tr><td>Model Name</td><td>{data.get('model_name', 'N/A')}</td></tr>
            <tr><td>Model Type</td><td>{data.get('model_type', 'N/A')}</td></tr>
            <tr><td>Risk Level</td><td>{data.get('risk_level', 'N/A')}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Confidence Required</td><td>{data.get('confidence_required', 'N/A')}</td></tr>
            <tr><td>Empirical Coverage</td><td>{data.get('empirical_coverage', 'N/A')}</td></tr>
            <tr><td>Attack Success Rate</td><td>{data.get('adversarial_metrics', {}).get('attack_success_rate', 'N/A')}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Security Assessment</h2>
        <table>
            <tr><th>Field</th><th>Value</th></tr>
            <tr><td>Attack Method</td><td>{data.get('adversarial_metrics', {}).get('attack_type', 'N/A')}</td></tr>
            <tr><td>Attack Success Rate</td><td>{data.get('adversarial_metrics', {}).get('attack_success_rate', 'N/A')}</td></tr>
            <tr><td>Samples Tested</td><td>{data.get('adversarial_metrics', {}).get('samples_tested', 'N/A')}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Data Drift Monitoring</h2>
        <table>
            <tr><th>Field</th><th>Value</th></tr>
            <tr><td>Drift Status</td><td>{data.get('data_drift_status', 'N/A')}</td></tr>
            <tr><td>Alert Required</td><td>{data.get('data_drift_alert', 'N/A')}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Audit Trail</h2>
        <table>
            <tr><th>Field</th><th>Value</th></tr>
            <tr><td>Lineage Run ID</td><td>{data.get('lineage_run_id', 'N/A')}</td></tr>
            <tr><td>Audit Log Path</td><td>{data.get('audit_log_path', 'N/A')}</td></tr>
        </table>
    </div>
</body>
</html>
"""

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Fallback HTML report generated: {output_path}")

        return html
