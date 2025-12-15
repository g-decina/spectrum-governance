"""
Regulatory Citation Tracking Module

Tracks citations to regulatory requirements, ensuring all compliance claims
in reports are properly attributed to authoritative sources.

This module:
- Loads citation databases from YAML
- Validates citation freshness (staleness checking)
- Formats citations for reports
- Tracks which citations are used in which reports (audit trail)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import yaml
from loguru import logger


@dataclass
class RegulatatoryCitation:
    """A citation to a regulatory source."""

    # Identification
    id: str  # e.g., "EU_AI_ACT_ART15"
    regulation: str  # e.g., "EU_AI_ACT"

    # Citation details
    source: str  # e.g., "Regulation (EU) 2024/1689"
    section: str  # e.g., "Article 15, Paragraph 3"
    url: str  # Official source URL
    text_excerpt: Optional[str] = None  # Relevant excerpt from regulation

    # Staleness tracking
    effective_date: datetime = field(default_factory=datetime.now)
    last_verified: datetime = field(default_factory=datetime.now)
    review_frequency_days: int = 180

    # Metadata
    hierarchy_level: int = 1  # 1=STATUTORY, 2=REGULATORY, 3=ACADEMIC, 4=INDUSTRY, 5=EXPERT
    hierarchy_name: str = "STATUTORY"
    jurisdiction: List[str] = field(default_factory=list)  # ["EU", "US", "UK"]

    def is_stale(self) -> bool:
        """Check if this citation needs re-verification."""
        staleness_threshold = timedelta(days=self.review_frequency_days)
        return (datetime.now() - self.last_verified) > staleness_threshold

    def format_citation(self, style: str = "legal") -> str:
        """
        Format citation for inclusion in reports.

        Args:
            style: "legal" (formal), "inline" (parenthetical), or "footnote"

        Returns:
            Formatted citation string
        """
        if style == "legal":
            return f"{self.source}, {self.section}"
        elif style == "inline":
            return f"({self.source} {self.section})"
        elif style == "footnote":
            return f"{self.source}, {self.section}. Available at: {self.url}"
        else:
            return f"{self.source}, {self.section}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "regulation": self.regulation,
            "source": self.source,
            "section": self.section,
            "url": self.url,
            "text_excerpt": self.text_excerpt,
            "effective_date": self.effective_date.isoformat(),
            "last_verified": self.last_verified.isoformat(),
            "review_frequency_days": self.review_frequency_days,
            "hierarchy_level": self.hierarchy_level,
            "hierarchy_name": self.hierarchy_name,
            "jurisdiction": self.jurisdiction
        }


class CitationTracker:
    """
    Manages regulatory citations and tracks their usage in reports.

    This provides an audit trail showing which regulatory requirements
    were cited in which reports, ensuring traceability.
    """

    def __init__(self, citation_database_path: Optional[str] = None):
        """
        Initialize citation tracker.

        Args:
            citation_database_path: Path to citation database YAML file
        """
        if citation_database_path:
            self.db_path = Path(citation_database_path)
        else:
            # Default to threshold_schema.yaml in same directory
            self.db_path = Path(__file__).parent / "threshold_schema.yaml"

        self.citations: Dict[str, RegulatatoryCitation] = {}
        self.usage_log: List[Dict] = []  # Track which reports use which citations

        self._load_citations()

    def _load_citations(self) -> None:
        """Load citations from YAML database."""
        if not self.db_path.exists():
            logger.warning(f"Citation database not found: {self.db_path}")
            return

        try:
            with open(self.db_path, 'r') as f:
                data = yaml.safe_load(f)

            # Extract citations from thresholds
            thresholds = data.get("thresholds", [])
            for threshold in thresholds:
                citation_data = threshold.get("citation", {})
                citation_id = threshold.get("id", "UNKNOWN")

                self.citations[citation_id] = RegulatatoryCitation(
                    id=citation_id,
                    regulation=threshold.get("regulation", "UNKNOWN"),
                    source=citation_data.get("source", ""),
                    section=citation_data.get("section", ""),
                    url=citation_data.get("url", ""),
                    text_excerpt=threshold.get("description"),
                    effective_date=datetime.fromisoformat(
                        citation_data.get("effective_date", datetime.now().isoformat())
                    ),
                    last_verified=datetime.fromisoformat(
                        citation_data.get("last_reviewed", datetime.now().isoformat())
                    ),
                    review_frequency_days=citation_data.get("review_frequency_days", 180),
                    hierarchy_level=threshold.get("hierarchy_level", 1),
                    hierarchy_name=threshold.get("hierarchy_name", "STATUTORY"),
                    jurisdiction=threshold.get("applicability", {}).get("jurisdictions", [])
                )

            logger.info(f"Loaded {len(self.citations)} regulatory citations")

        except Exception as e:
            logger.error(f"Failed to load citations: {e}")

    def get_citation(self, citation_id: str) -> Optional[RegulatatoryCitation]:
        """Get a specific citation by ID."""
        return self.citations.get(citation_id)

    def get_citations_by_regulation(self, regulation: str) -> List[RegulatatoryCitation]:
        """Get all citations for a specific regulation."""
        return [
            cite for cite in self.citations.values()
            if cite.regulation == regulation
        ]

    def get_stale_citations(self) -> List[RegulatatoryCitation]:
        """Get all citations that need re-verification."""
        return [cite for cite in self.citations.values() if cite.is_stale()]

    def log_citation_usage(
        self,
        citation_id: str,
        report_path: str,
        context: str
    ) -> None:
        """
        Log that a citation was used in a report.

        This creates an audit trail showing which regulatory requirements
        were cited in which reports.

        Args:
            citation_id: ID of the citation
            report_path: Path to the report file
            context: Where in the report the citation was used
        """
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "citation_id": citation_id,
            "report_path": report_path,
            "context": context
        }

        self.usage_log.append(usage_entry)
        logger.debug(f"Logged citation usage: {citation_id} in {report_path}")

    def generate_citation_report(self) -> dict:
        """
        Generate a report of all citations and their usage.

        Returns:
            Dictionary with citation statistics and usage patterns
        """
        stale_citations = self.get_stale_citations()

        # Group citations by regulation
        by_regulation = {}
        for cite in self.citations.values():
            if cite.regulation not in by_regulation:
                by_regulation[cite.regulation] = []
            by_regulation[cite.regulation].append(cite.id)

        # Count citation usage
        usage_counts = {}
        for usage in self.usage_log:
            cit_id = usage["citation_id"]
            usage_counts[cit_id] = usage_counts.get(cit_id, 0) + 1

        return {
            "total_citations": len(self.citations),
            "stale_citations": len(stale_citations),
            "citations_by_regulation": by_regulation,
            "usage_counts": usage_counts,
            "stale_citation_ids": [c.id for c in stale_citations],
            "most_used_citations": sorted(
                usage_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def format_bibliography(
        self,
        citation_ids: List[str],
        style: str = "legal"
    ) -> str:
        """
        Format a bibliography section for a report.

        Args:
            citation_ids: List of citation IDs to include
            style: Citation style ("legal", "inline", "footnote")

        Returns:
            Formatted bibliography as a string
        """
        bibliography = []

        for cit_id in citation_ids:
            citation = self.citations.get(cit_id)
            if citation:
                formatted = citation.format_citation(style)
                bibliography.append(f"[{cit_id}] {formatted}")

        return "\n".join(bibliography)

    def validate_citation_freshness(self) -> Dict[str, List[str]]:
        """
        Validate that all citations are fresh (not stale).

        Returns:
            Dictionary with "fresh" and "stale" lists of citation IDs
        """
        fresh = []
        stale = []

        for cit_id, citation in self.citations.items():
            if citation.is_stale():
                stale.append(cit_id)
            else:
                fresh.append(cit_id)

        return {
            "fresh": fresh,
            "stale": stale,
            "fresh_count": len(fresh),
            "stale_count": len(stale),
            "total": len(self.citations)
        }
