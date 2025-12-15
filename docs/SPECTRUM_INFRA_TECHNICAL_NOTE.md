# Spectrum Infrastructure Technical Note

## Core Types, Logging, and Privacy Framework

**Version:** 1.0.0
**Status:** Production Ready

---

## Overview

`spectrum.infra` provides the foundational infrastructure for the Spectrum governance framework. This module defines core types, privacy policies, and logging mechanisms that ensure consistency across all Spectrum modules.

```
spectrum.infra/
├── types.py    # Core type definitions (RiskProfile, InferenceEvent, TargetModel)
├── privacy.py  # PII detection and sanitization (PIIDetector, PIISanitizer)
├── logger.py   # RCIA audit logging with integrity hashing (RCIALogger)
└── events.py   # Event system for cross-module communication
```

---

## Core Types

### RiskProfile

The immutable governance contract that encapsulates the coverage guarantee:

$$P(Y \in \hat{C}(X)) \geq 1 - \alpha$$

```python
from spectrum.infra.types import RiskProfile, RiskLevel

# High-risk credit decision (95% confidence)
profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)

# Critical autonomous system (99% confidence)
profile = RiskProfile(level=RiskLevel.CRITICAL, alpha=0.01)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `level` | `RiskLevel` | Risk classification (LOW, MEDIUM, HIGH, CRITICAL) |
| `alpha` | `float` | Significance level (1 - confidence), range (0, 1) |
| `pii_policy` | `PIIPolicy` | Auto-assigned based on risk level |

**Risk Level Mappings:**

| Level | Alpha Range | PII Policy | Use Case |
|-------|-------------|------------|----------|
| LOW | 0.20 | DETECT_ONLY | Marketing, A/B testing |
| MEDIUM | 0.10 | DETECT_ONLY | Fraud alerts |
| HIGH | 0.05 | PSEUDONYMIZE | Credit decisions |
| CRITICAL | 0.01 | REDACT | Autonomous vehicles |

**Immutability:**

RiskProfile is frozen (Pydantic `frozen=True`), ensuring governance contracts cannot be accidentally modified:

```python
profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)
profile.alpha = 0.10  # Raises: FrozenInstanceError
```

### InferenceEvent

The audit log atom that captures every model interaction:

```python
from spectrum.infra.types import InferenceEvent
import numpy as np

event = InferenceEvent(
    model_version="v2.1.0",
    input_payload=np.array([[1.0, 2.0, 3.0]]),
    output_payload={"prediction": 1, "probability": 0.87}
)

print(event.event_id)      # uuid4
print(event.timestamp)     # datetime
print(event.data_type)     # "numpy"
print(event.input_shape)   # [1, 3]
print(event.input_bytes)   # 24
```

**Automatic Metadata Extraction:**

The `InferenceEvent` automatically computes metadata from payloads:

| Payload Type | Detected As | Shape | Bytes |
|--------------|-------------|-------|-------|
| `np.ndarray` | "numpy" | array.shape | array.nbytes |
| `pd.DataFrame` | "pandas" | df.shape | memory_usage(deep=True) |
| `torch.Tensor` | "torch" | tensor.shape | element_size * nelement |
| `str` | "string" | [len(str)] | sys.getsizeof |

### TargetModel

Unified contract for model specification with mutual exclusivity enforcement:

```python
from spectrum.infra.types import TargetModel
from sklearn.ensemble import RandomForestClassifier

# Local sklearn model
target = TargetModel(model_object=trained_model)
assert target.is_tabular == True

# Remote LLM API
target = TargetModel(target_api_url="https://api.openai.com/v1/chat/completions")
assert target.is_tabular == False

# Invalid: Both specified
target = TargetModel(model_object=model, target_api_url="...") # Raises ValueError
```

---

## Privacy Framework

### PIIPolicy

Policy-driven PII handling with four levels:

```python
from spectrum.infra.privacy import PIIPolicy

PIIPolicy.DETECT_ONLY   # Warn but don't modify (safe default)
PIIPolicy.PSEUDONYMIZE  # SHA-256 hash with salt
PIIPolicy.REDACT        # Remove PII fields entirely
PIIPolicy.ALLOW         # Skip PII checks (explicit opt-out)
```

### PIIDetector

Automatic PII detection using multiple strategies:

```python
from spectrum.infra.privacy import PIIDetector, PIISchema
import pandas as pd

# Automatic detection
detector = PIIDetector()
df = pd.DataFrame({
    "email": ["user@example.com", "test@test.com"],
    "ssn": ["123-45-6789", "987-65-4321"],
    "income": [50000, 75000]
})

results = detector.detect(df)
for r in results:
    print(f"{r.field_name}: {r.pii_type.value} (confidence: {r.confidence:.0%})")
# email: contact (confidence: 95%)
# ssn: identifier (confidence: 95%)
```

**Detection Methods:**

1. **Column Name Matching**
   - Recognizes common PII field names: `email`, `ssn`, `phone`, `address`, etc.
   - Confidence: 85%

2. **Regex Pattern Matching**
   - Email: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
   - Phone: `^\+?1?\d{9,15}$` or `^\(\d{3}\)\s?\d{3}-?\d{4}$`
   - SSN: `^\d{3}-?\d{2}-?\d{4}$`
   - Credit Card: `^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$`
   - IP Address: `^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$`
   - Confidence: 95% (if 80%+ samples match)

3. **Cardinality Analysis**
   - Flags high-cardinality string fields (>80% unique, >100 values)
   - Indicates potential unique identifiers
   - Confidence: 70%

**Explicit Schema Override:**

```python
# Override automatic detection with explicit schema
schema = PIISchema(
    pii_fields={
        "customer_id": PIIFieldType.IDENTIFIER,
        "medical_notes": PIIFieldType.SENSITIVE
    },
    safe_fields={"product_category", "transaction_date"}
)

detector = PIIDetector(schema=schema)
results = detector.detect(df)
```

### PIISanitizer

Log/payload sanitization for audit trails:

```python
from spectrum.infra.privacy import PIISanitizer

sanitizer = PIISanitizer()

# Sanitize payload
payload = {
    "email": "user@example.com",
    "ssn": "123-45-6789",
    "amount": 1000
}

safe_payload = sanitizer.sanitize(payload)
# {"email": "[EMAIL]", "ssn": "[SSN]", "amount": 1000}
```

**Pattern Replacements:**

| Pattern | Replacement | Regex |
|---------|-------------|-------|
| Email | `[EMAIL]` | `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` |
| SSN | `[SSN]` | `[0-9]{3}-[0-9]{2}-[0-9]{4}` or `[0-9]{9}` |
| Phone | `[PHONE]` | Various formats |

**Vector Summarization:**

Large arrays and DataFrames are summarized to prevent log bloat:

```python
import numpy as np
import pandas as pd

sanitizer.sanitize(np.array([[1, 2], [3, 4]]))
# "<ndarray shape=(2, 2) dtype=float64>"

sanitizer.sanitize(pd.DataFrame({"a": [1, 2, 3]}))
# "<Pandas DataFrame shape=(3, 1)>"
```

### PIIDataSanitizer

Full DataFrame sanitization for ML pipelines:

```python
from spectrum.infra.privacy import PIIDataSanitizer, PIIPolicy

# Pseudonymization (hash PII, preserve uniqueness)
sanitizer = PIIDataSanitizer(policy=PIIPolicy.PSEUDONYMIZE)
df_clean, audit = sanitizer.sanitize(df, detection_results)

print(audit)
# {"policy": "pseudonymize", "fields_modified": ["email", "ssn"], "n_fields": 2}

# Redaction (remove PII entirely)
sanitizer = PIIDataSanitizer(policy=PIIPolicy.REDACT)
df_clean, audit = sanitizer.sanitize(df, detection_results)
# df_clean no longer contains email or ssn columns
```

**Pseudonymization Algorithm:**

```python
# SHA-256 with salt for deterministic but irreversible mapping
salted = f"{salt}:{value}"
hash_obj = hashlib.sha256(salted.encode())
pseudonymized = hash_obj.hexdigest()[:16]  # First 16 chars
```

---

## RCIA Logging

### RCIALogger

Risk, Compliance, Inference, and Audit logging with integrity guarantees:

```python
from spectrum.infra.logger import RCIALogger
from spectrum.infra.types import InferenceEvent

logger = RCIALogger(log_path="audit.jsonl")

event = InferenceEvent(
    model_version="v2.1.0",
    input_payload=X_test[0],
    output_payload={"prediction": 1}
)

logger.log_event(event, context="INFERENCE")
```

**Log Entry Structure:**

```json
{
  "rcia_context": "INFERENCE",
  "audit_hash": "sha256:a1b2c3d4e5f6...",
  "data": {
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-01-15T10:30:00.000000",
    "model_version": "v2.1.0",
    "input_payload_sanitized": "<ndarray shape=(1, 15) dtype=float64>",
    "output_payload_sanitized": {"prediction": 1},
    "input_shape": [1, 15],
    "input_bytes": 120,
    "data_type": "numpy"
  }
}
```

**Key Features:**

1. **Integrity Hashing**
   ```python
   # Each entry is hashed for tamper detection
   json_data = json.dumps(log_entry, sort_keys=True)
   audit_hash = hashlib.sha256(json_data.encode()).hexdigest()
   ```

2. **Non-blocking I/O**
   ```python
   # Loguru with enqueue=True for async writes
   logger.add(self.log_path, enqueue=True)
   ```

3. **Automatic Rotation**
   ```python
   # 10MB rotation with compression
   logger.add(self.log_path, rotation="10 MB", compression="zip")
   ```

4. **PII Sanitization**
   ```python
   # Automatic sanitization before logging
   log_entry['input_payload_sanitized'] = self.sanitizer.sanitize(payload)
   ```

---

## Deep Learning Adapter

### DLAdapter

Sklearn-compatible wrapper for PyTorch and TensorFlow models:

```python
from spectrum.utils.dl_adapter import DLAdapter
import torch

# Wrap PyTorch model for sklearn compatibility
class MyModel(torch.nn.Module):
    ...

dl_adapter = DLAdapter(my_pytorch_model)

# Now works with sklearn-style predict()
predictions = dl_adapter.predict(X_test)

# And with Spectrum wrappers
trusted = SpectrumUncertaintyWrapper(
    base_model=dl_adapter,  # Wrapped DL model
    risk_profile=profile
)
```

**Detection:**

```python
adapter = DLAdapter(model)
if adapter.is_torch:
    # PyTorch model detected
elif adapter.is_tf:
    # TensorFlow model detected
```

---

## API Reference

### RiskLevel Enum

```python
class RiskLevel(str, Enum):
    LOW = "Low"       # Marketing segmentation
    MEDIUM = "Medium" # Fraud alert
    HIGH = "High"     # Credit denial
    CRITICAL = "Critical"  # Autonomous braking
```

### RiskProfile Model

```python
class RiskProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: RiskLevel
    alpha: float = Field(..., gt=0.0, lt=1.0)
    pii_policy: Optional[PIIPolicy] = None  # Auto-set if not provided

    @field_validator("alpha")
    def warn_if_loose(cls, v: float) -> float:
        if v < 0.01:
            warnings.warn(f"Alpha {v} is extremely low.")
        return v
```

### InferenceEvent Model

```python
class InferenceEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str

    input_payload: Any = None
    output_payload: Any = None

    # Computed automatically
    input_shape: Optional[List[int]] = None
    input_bytes: Optional[int] = None
    data_type: str = "unknown"
```

### PIIFieldType Enum

```python
class PIIFieldType(str, Enum):
    IDENTIFIER = "identifier"           # SSN, email (direct identifier)
    QUASI_IDENTIFIER = "quasi_identifier"  # ZIP, age (indirect)
    CONTACT = "contact"                 # Phone, address
    SENSITIVE = "sensitive"             # Health, financial
    BIOMETRIC = "biometric"             # Fingerprint, face
    SAFE = "safe"                       # Explicitly non-PII
```

### PIIDetectionResult

```python
@dataclass
class PIIDetectionResult:
    field_name: str
    pii_type: PIIFieldType
    confidence: float  # 0.0-1.0
    reason: str
    sample_values: Optional[List[str]] = None
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPECTRUM_LOG_LEVEL` | `INFO` | Logging verbosity |
| `SPECTRUM_AUDIT_PATH` | `rcia_audit.jsonl` | Default audit log path |
| `SPECTRUM_PII_POLICY` | `detect` | Default PII policy |

### Loguru Configuration

```python
from loguru import logger

# Default configuration applied by RCIALogger
logger.add(
    log_path,
    serialize=False,      # Raw JSON (not loguru format)
    enqueue=True,         # Async writes
    rotation="10 MB",     # Rotate at 10MB
    compression="zip",    # Compress old logs
    level="INFO"
)
```

---

## Security Considerations

### Hash Integrity

```python
# Verify log entry integrity
import hashlib
import json

with open("audit.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        data = entry["data"]
        json_data = json.dumps(data, sort_keys=True)
        computed_hash = hashlib.sha256(json_data.encode()).hexdigest()

        if computed_hash != entry["audit_hash"]:
            raise SecurityError(f"Log tampering detected: {entry['data']['event_id']}")
```

### PII Salt Management

```python
# Securely manage pseudonymization salt
import os

# Load from environment (recommended)
salt = os.environ.get("SPECTRUM_PII_SALT")

# Or generate securely
import secrets
salt = secrets.token_hex(16)

sanitizer = PIIDataSanitizer(policy=PIIPolicy.PSEUDONYMIZE, salt=salt)
```

### Minimum Privileges

```python
# RCIALogger only needs write access to log file
# No network access required
# No access to raw training data needed
```

---

## Best Practices

### RiskProfile Selection

1. **Start conservative** - Use higher confidence levels initially
2. **Domain alignment** - Match risk level to business consequences
3. **Regulatory requirements** - Some regulations mandate specific levels
4. **Document rationale** - Record why a specific level was chosen

### PII Handling

1. **Schema over detection** - Explicit schemas are more reliable
2. **Audit the audit** - Log PII handling decisions
3. **Retention limits** - Don't keep sanitized data indefinitely
4. **Access control** - Limit access to audit logs

### Logging Strategy

1. **Centralize logs** - Use a single RCIA log per deployment
2. **Monitor rotation** - Ensure sufficient disk space
3. **Backup hashes** - Store hash digests separately for verification
4. **Regular verification** - Periodically verify log integrity

---

## Example: Complete Infrastructure Setup

```python
from spectrum.infra.types import RiskProfile, RiskLevel, InferenceEvent
from spectrum.infra.logger import RCIALogger
from spectrum.infra.privacy import (
    PIIDetector, PIIDataSanitizer, PIIPolicy, PIISchema, PIIFieldType
)
import pandas as pd
import numpy as np

# 1. Define risk profile
profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)
print(f"PII Policy: {profile.pii_policy}")  # PSEUDONYMIZE

# 2. Setup PII detection with explicit schema
schema = PIISchema(
    pii_fields={
        "customer_id": PIIFieldType.IDENTIFIER,
        "email": PIIFieldType.CONTACT
    },
    safe_fields={"product_sku", "order_date"}
)

detector = PIIDetector(schema=schema)

# 3. Detect PII in data
df = pd.read_csv("transactions.csv")
pii_results = detector.detect(df)

print("Detected PII:")
for r in pii_results:
    print(f"  {r.field_name}: {r.pii_type.value}")

# 4. Sanitize data according to risk profile
sanitizer = PIIDataSanitizer(policy=profile.pii_policy)
df_clean, audit = sanitizer.sanitize(df, pii_results)

print(f"Sanitization: {audit}")

# 5. Setup audit logging
logger = RCIALogger(log_path="/var/log/spectrum/audit.jsonl")

# 6. Log inference events
event = InferenceEvent(
    model_version="v2.1.0",
    input_payload=df_clean.iloc[0].values,
    output_payload={"prediction": 1, "confidence": 0.95}
)

logger.log_event(event, context="PRODUCTION_INFERENCE")

print("Infrastructure configured successfully")
```

---

## References

1. **Pydantic**: https://docs.pydantic.dev/

2. **Loguru**: https://loguru.readthedocs.io/

3. **GDPR Article 25**: Data Protection by Design and by Default

4. **CCPA**: California Consumer Privacy Act

5. **SHA-256**: FIPS 180-4

6. **UUID**: RFC 4122

---

*Last updated: 2024*
