"""
spectrum.infra.privacy
======================

Privacy and PII (Personally Identifiable Information) Protection Framework.

This module provides comprehensive privacy capabilities:
1. Log/Payload Sanitization: Safe logging without exposing PII
2. DataFrame PII Detection: Automatic identification of sensitive fields
3. Data Sanitization: Policy-driven PII handling for ML pipelines

CORE PHILOSOPHY:
----------------
- Privacy by Design: GDPR Article 25 compliance
- Detect by Default: Automatically identify potential PII
- Risk-Aware: Action depends on RiskProfile/RiskLevel
- Transparent: Make PII handling explicit and auditable
- Flexible: Support schema-based and automatic detection

PRIVACY POLICIES:
-----------------
- DETECT_ONLY: Warn but don't modify (safe default)
- PSEUDONYMIZE: Hash/tokenize PII (preserve utility)
- REDACT: Remove PII fields entirely (maximum privacy)
- ALLOW: Skip PII checks (explicit opt-out)

USAGE:
------
# Log Sanitization:
log_sanitizer = PIISanitizer()
safe_payload = log_sanitizer.sanitize({"email": "user@example.com"})
# Returns: {"email": "[EMAIL]"}

# DataFrame PII Detection:
detector = PIIDetector()
results = detector.detect(dataframe)

# Data Sanitization:
data_sanitizer = PIIDataSanitizer(policy=PIIPolicy.PSEUDONYMIZE)
df_clean, audit = data_sanitizer.sanitize(df, results)
"""

import re
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional, Tuple
import pandas as pd
import numpy as np


# ============================================================================
# Log/Payload Sanitization (Existing)
# ============================================================================

class PIISanitizer:
    """
    Sanitizes PII from logs and JSON payloads for safe audit trails.

    Replaces PII patterns (EMAIL, SSN, PHONE) with placeholders like [EMAIL].
    Also summarizes large vectors/DataFrames to prevent log bloat.
    """

    STATIC_PATTERNS = {
        "EMAIL": [
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        ],
        "SSN": [
            re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
            re.compile(r"\b[0-9]{9}\b")
        ],
        "PHONE": [
            re.compile(r"\([0-9]{3}\)-[0-9]{3}-[0-9]{4}"),
            re.compile(r"[0-9]{3}-[0-9]{3}-[0-9]{4}"),
            re.compile(r"[0-9]{10}")
        ]
    }

    def sanitize(self, payload: Any) -> Any:
        """
        Recursively sanitizes a payload.
        Guarantees that the output is JSON-serializable.
        """
        if isinstance(payload, dict):
            return {k: self.sanitize(v) for k, v in payload.items()}

        if isinstance(payload, list):
            return [self.sanitize(item) for item in payload]

        if isinstance(payload, str):
            for cat, patterns in self.STATIC_PATTERNS.items():
                for pattern in patterns:
                    payload = re.sub(pattern, f"[{cat}]", payload)
            return payload

        if isinstance(payload, (int, float, bool, type(None))):
            return payload

        return self._summarize_vector(payload)

    def _summarize_vector(self, data: Any) -> str:
        """
        Prevents large vectors from entering the log stream.
        """
        t_name = type(data).__name__

        if "Tensor" in t_name or "ndarray" in t_name:
            shape = getattr(data, "shape", "unknown")
            dtype = getattr(data, "dtype", "unknown")
            return f"<{t_name} shape={shape} dtype={dtype}>"

        if "DataFrame" in t_name:
            shape = getattr(data, "shape", "unknown")
            return f"<Pandas DataFrame shape={shape}>"

        # Fallback for arbitrary objects (e.g. Model Classes)
        return f"<Object: {t_name}>"


# ============================================================================
# DataFrame PII Detection & Sanitization (New)
# ============================================================================

class PIIPolicy(str, Enum):
    """
    PII handling policy determining action when PII is detected.
    """
    DETECT_ONLY = "detect"          # Warn but don't modify data
    PSEUDONYMIZE = "pseudonymize"   # Hash/tokenize PII fields
    REDACT = "redact"               # Remove PII fields entirely
    ALLOW = "allow"                 # Skip PII checks (explicit opt-out)


class PIIFieldType(str, Enum):
    """
    Classification of PII field types for regulatory compliance.
    """
    IDENTIFIER = "identifier"                   # Direct identifier (SSN, email)
    QUASI_IDENTIFIER = "quasi_identifier"       # Indirect identifier (ZIP, age)
    CONTACT = "contact"                         # Contact info (phone, email, address)
    SENSITIVE = "sensitive"                     # Sensitive data (health, financial)
    BIOMETRIC = "biometric"                     # Biometric data (fingerprint, face)
    SAFE = "safe"                               # Explicitly non-PII


@dataclass
class PIIDetectionResult:
    """
    Results from automatic PII detection.
    """
    field_name: str                             # Column/field name
    pii_type: PIIFieldType                      # Detected PII type
    confidence: float                           # Detection confidence (0.0-1.0)
    reason: str                                 # Why this was flagged as PII
    sample_values: Optional[List[str]] = None   # Sample values (for inspection)

    def __repr__(self) -> str:
        return (
            f"PIIDetectionResult(field='{self.field_name}', "
            f"type={self.pii_type.value}, confidence={self.confidence:.2f}, "
            f"reason='{self.reason}')"
        )


@dataclass
class PIISchema:
    """
    Explicit PII schema definition for data governance.

    Allows users to explicitly mark fields as PII or safe,
    overriding automatic detection.
    """
    pii_fields: Dict[str, PIIFieldType] = field(default_factory=dict)
    safe_fields: Set[str] = field(default_factory=set)

    def is_pii(self, field_name: str) -> bool:
        """Check if field is explicitly marked as PII."""
        return field_name in self.pii_fields

    def is_safe(self, field_name: str) -> bool:
        """Check if field is explicitly marked as safe."""
        return field_name in self.safe_fields

    def get_pii_type(self, field_name: str) -> Optional[PIIFieldType]:
        """Get PII type for field, or None if not PII."""
        return self.pii_fields.get(field_name)


class PIIDetector:
    """
    Automatic PII detection using pattern matching, heuristics, and statistical analysis.

    Detection Methods:
    ------------------
    1. Column Name Matching: Common PII field names
    2. Regex Patterns: Email, phone, SSN, credit card patterns
    3. Cardinality Analysis: High cardinality string fields (potential IDs)
    4. Entropy Analysis: Random vs structured data
    """

    # Common PII field names (lowercase)
    PII_FIELD_NAMES = {
        PIIFieldType.IDENTIFIER: {
            'ssn', 'social_security', 'social_security_number', 'ein',
            'passport', 'license', 'customer_id', 'user_id', 'patient_id',
            'account_number', 'member_id', 'employee_id'
        },
        PIIFieldType.CONTACT: {
            'email', 'e_mail', 'mail', 'phone', 'telephone', 'mobile',
            'address', 'street', 'city', 'zip', 'zipcode', 'postal',
            'ip_address', 'ip', 'url', 'website'
        },
        PIIFieldType.IDENTIFIER: {
            'name', 'first_name', 'last_name', 'full_name', 'firstname',
            'lastname', 'username', 'login', 'nickname'
        },
        PIIFieldType.SENSITIVE: {
            'dob', 'date_of_birth', 'birthdate', 'age', 'gender', 'sex',
            'race', 'ethnicity', 'religion', 'nationality',
            'salary', 'income', 'credit_score', 'balance'
        }
    }

    # Regex patterns for PII detection
    REGEX_PATTERNS = {
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'phone': re.compile(r'^\+?1?\d{9,15}$|^\(\d{3}\)\s?\d{3}-?\d{4}$'),
        'ssn': re.compile(r'^\d{3}-?\d{2}-?\d{4}$'),
        'credit_card': re.compile(r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'),
        'ip_address': re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'),
        'zip_code': re.compile(r'^\d{5}(-\d{4})?$'),
    }

    def __init__(self, schema: Optional[PIISchema] = None):
        """
        Initialize PII detector.

        :param schema: Optional explicit schema to override detection
        """
        self.schema = schema or PIISchema()

    def detect(self, df: pd.DataFrame, sample_size: int = 100) -> List[PIIDetectionResult]:
        """
        Detect PII in a pandas DataFrame.

        :param df: Input DataFrame to scan
        :param sample_size: Number of rows to sample for pattern detection
        :return: List of PII detection results
        """
        results = []

        for column in df.columns:
            # Skip if explicitly marked as safe
            if self.schema.is_safe(column):
                continue

            # Use explicit schema if available
            if self.schema.is_pii(column):
                pii_type = self.schema.get_pii_type(column)
                results.append(PIIDetectionResult(
                    field_name=column,
                    pii_type=pii_type,
                    confidence=1.0,
                    reason="Explicit schema definition"
                ))
                continue

            # Automatic detection
            detection = self._detect_field(df[column], column, sample_size)
            if detection:
                results.append(detection)

        return results

    def _detect_field(
        self,
        series: pd.Series,
        field_name: str,
        sample_size: int
    ) -> Optional[PIIDetectionResult]:
        """
        Detect if a single field contains PII.

        :param series: Pandas series to analyze
        :param field_name: Name of the field
        :param sample_size: Number of values to sample
        :return: PIIDetectionResult if PII detected, None otherwise
        """
        # Method 1: Column name matching
        name_result = self._check_field_name(field_name)
        if name_result:
            return name_result

        # Method 2: Pattern matching (for string fields)
        if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
            pattern_result = self._check_patterns(series, field_name, sample_size)
            if pattern_result:
                return pattern_result

            # Method 3: Cardinality analysis (potential identifiers)
            cardinality_result = self._check_cardinality(series, field_name)
            if cardinality_result:
                return cardinality_result

        return None

    def _check_field_name(self, field_name: str) -> Optional[PIIDetectionResult]:
        """Check if field name matches common PII patterns."""
        field_lower = field_name.lower().replace('_', '').replace('-', '')

        for pii_type, keywords in self.PII_FIELD_NAMES.items():
            for keyword in keywords:
                keyword_clean = keyword.replace('_', '').replace('-', '')
                if keyword_clean in field_lower or field_lower in keyword_clean:
                    return PIIDetectionResult(
                        field_name=field_name,
                        pii_type=pii_type,
                        confidence=0.85,
                        reason=f"Field name matches PII keyword: '{keyword}'"
                    )

        return None

    def _check_patterns(
        self,
        series: pd.Series,
        field_name: str,
        sample_size: int
    ) -> Optional[PIIDetectionResult]:
        """Check if field values match PII regex patterns."""
        # Sample non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return None

        sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)
        sample_str = sample.astype(str)

        for pattern_name, regex in self.REGEX_PATTERNS.items():
            matches = sample_str.apply(lambda x: bool(regex.match(x)))
            match_ratio = matches.sum() / len(sample_str)

            if match_ratio >= 0.8:  # 80% of samples match pattern
                pii_type_map = {
                    'email': PIIFieldType.CONTACT,
                    'phone': PIIFieldType.CONTACT,
                    'ssn': PIIFieldType.IDENTIFIER,
                    'credit_card': PIIFieldType.SENSITIVE,
                    'ip_address': PIIFieldType.CONTACT,
                    'zip_code': PIIFieldType.QUASI_IDENTIFIER,
                }

                return PIIDetectionResult(
                    field_name=field_name,
                    pii_type=pii_type_map.get(pattern_name, PIIFieldType.IDENTIFIER),
                    confidence=min(0.95, match_ratio),
                    reason=f"Values match {pattern_name} pattern ({match_ratio:.0%} match rate)",
                    sample_values=sample_str.head(3).tolist()
                )

        return None

    def _check_cardinality(
        self,
        series: pd.Series,
        field_name: str
    ) -> Optional[PIIDetectionResult]:
        """Check for high-cardinality fields that might be identifiers."""
        n_unique = series.nunique()
        n_total = len(series)

        # High cardinality ratio suggests unique identifiers
        cardinality_ratio = n_unique / n_total if n_total > 0 else 0

        # Flag if >80% unique and >100 unique values
        if cardinality_ratio > 0.8 and n_unique > 100:
            return PIIDetectionResult(
                field_name=field_name,
                pii_type=PIIFieldType.IDENTIFIER,
                confidence=0.70,
                reason=f"High cardinality: {n_unique:,} unique values ({cardinality_ratio:.0%} unique)"
            )

        return None


class PIIDataSanitizer:
    """
    PII sanitization for DataFrames through pseudonymization or redaction.

    Sanitization Methods:
    ---------------------
    1. Pseudonymization: SHA-256 hashing with salt (preserves uniqueness)
    2. Redaction: Complete removal of PII fields
    3. Audit Logging: Track all PII handling operations
    """

    def __init__(self, policy: PIIPolicy, salt: Optional[str] = None):
        """
        Initialize PII data sanitizer.

        :param policy: PII handling policy
        :param salt: Salt for pseudonymization (random if not provided)
        """
        self.policy = policy
        self.salt = salt or self._generate_salt()
        self.audit_log: List[Dict] = []

    @staticmethod
    def _generate_salt() -> str:
        """Generate a random salt for hashing."""
        import secrets
        return secrets.token_hex(16)

    def sanitize(
        self,
        df: pd.DataFrame,
        detection_results: List[PIIDetectionResult]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Sanitize PII in DataFrame according to policy.

        :param df: Input DataFrame
        :param detection_results: PII detection results
        :return: (Sanitized DataFrame, Audit report)
        """
        if self.policy == PIIPolicy.ALLOW or self.policy == PIIPolicy.DETECT_ONLY:
            # No modification
            return df.copy(), {
                'policy': self.policy.value,
                'fields_modified': [],
                'action': 'none'
            }

        df_sanitized = df.copy()
        fields_modified = []

        for result in detection_results:
            field_name = result.field_name

            if field_name not in df_sanitized.columns:
                continue

            if self.policy == PIIPolicy.PSEUDONYMIZE:
                # Pseudonymize using hash
                df_sanitized[field_name] = df_sanitized[field_name].apply(
                    lambda x: self._pseudonymize_value(x) if pd.notna(x) else x
                )
                action = 'pseudonymized'

            elif self.policy == PIIPolicy.REDACT:
                # Remove field entirely
                df_sanitized = df_sanitized.drop(columns=[field_name])
                action = 'redacted'

            fields_modified.append(field_name)

            # Audit log
            self.audit_log.append({
                'field': field_name,
                'pii_type': result.pii_type.value,
                'action': action,
                'policy': self.policy.value
            })

        audit_report = {
            'policy': self.policy.value,
            'fields_modified': fields_modified,
            'action': 'pseudonymize' if self.policy == PIIPolicy.PSEUDONYMIZE else 'redact',
            'n_fields': len(fields_modified)
        }

        return df_sanitized, audit_report

    def _pseudonymize_value(self, value) -> str:
        """Pseudonymize a single value using SHA-256 hash."""
        value_str = str(value)
        salted = f"{self.salt}:{value_str}"
        hash_obj = hashlib.sha256(salted.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars of hash

    def get_audit_log(self) -> List[Dict]:
        """Get audit log of all sanitization operations."""
        return self.audit_log.copy()


def get_recommended_policy(risk_level: str) -> PIIPolicy:
    """
    Get recommended PII policy based on risk level.

    :param risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
    :return: Recommended PII policy
    """
    from spectrum.infra.types import RiskLevel

    recommendations = {
        RiskLevel.LOW: PIIPolicy.DETECT_ONLY,
        RiskLevel.MEDIUM: PIIPolicy.DETECT_ONLY,
        RiskLevel.HIGH: PIIPolicy.PSEUDONYMIZE,
        RiskLevel.CRITICAL: PIIPolicy.REDACT,
    }

    try:
        level = RiskLevel[risk_level.upper()]
        return recommendations[level]
    except (KeyError, AttributeError):
        return PIIPolicy.DETECT_ONLY  # Safe default
