# Regulatory Thresholds Quick Reference

**Purpose:** Quick reference for using the new regulatory thresholds system.

---

## Available Regulatory Contexts

### 1. EU AI Act
**Context ID:** `EU_AI_ACT`

**Applicable Thresholds:**
| Metric | Threshold | Comparison | Citation |
|--------|-----------|------------|----------|
| attack_success_rate | 0.25 | < | EU AI Act Article 15 |
| empirical_robustness_l2 | 0.5 | < | EU AI Act Annex IV |
| empirical_coverage | 0.95 | >= | EU AI Act Article 15 |

**Example Usage:**
```python
from spectrum.blue.thresholds import get_thresholds_for_context

thresholds = get_thresholds_for_context("EU_AI_ACT")
for t in thresholds:
    print(f"{t.metric_name} {t.comparison} {t.threshold_value}")
```

---

### 2. CFPB (Consumer Financial Protection Bureau)
**Context ID:** `CFPB`

**Applicable Thresholds:**
| Metric | Threshold | Comparison | Citation |
|--------|-----------|------------|----------|
| attack_success_rate | 0.20 | < | CFPB Bulletin 2023-02 |
| empirical_coverage | 0.90 | >= | SR 11-7 |
| max_psi | 0.25 | < | CFPB Circular 2022-03 |

**Use Case:** Credit scoring, lending models, fair lending compliance

---

### 3. EEOC (Equal Employment Opportunity Commission)
**Context ID:** `EEOC`

**Applicable Thresholds:**
| Metric | Threshold | Comparison | Citation |
|--------|-----------|------------|----------|
| disparate_impact_ratio | 0.80 | >= | 29 CFR 1607.4(D) |
| attack_success_rate | 0.30 | < | EEOC AI Guidance 2023 |

**Use Case:** Employment screening, hiring algorithms

---

### 4. NIST AI Risk Management Framework
**Context ID:** `NIST_AI_RMF`

**Applicable Thresholds:**
| Metric | Threshold | Comparison | Citation |
|--------|-----------|------------|----------|
| attack_success_rate | 0.15 | < | NIST AI RMF 1.0: MEASURE 2.3 |
| empirical_coverage | 0.95 | >= | NIST AI RMF 1.0: MEASURE 2.10 |

**Use Case:** General AI risk management, federal compliance

---

### 5. ISO/IEC 42001
**Context ID:** `ISO_IEC_42001`

**Applicable Thresholds:**
| Metric | Threshold | Comparison | Citation |
|--------|-----------|------------|----------|
| attack_success_rate | 0.20 | < | ISO/IEC 42001:2023 |

**Use Case:** AI management system certification

---

## Usage Examples

### Basic Threshold Lookup

```python
from spectrum.blue.thresholds import get_threshold_by_metric

# Find the most stringent threshold for attack_success_rate
threshold = get_threshold_by_metric("attack_success_rate", "EU_AI_ACT,CFPB")

print(f"Metric: {threshold.metric_name}")
print(f"Must be {threshold.comparison} {threshold.threshold_value}")
print(f"Citation: {threshold.citation}")
```

### Evaluate Against Thresholds

```python
from spectrum.blue.thresholds import get_threshold_by_metric

threshold = get_threshold_by_metric("attack_success_rate", "CFPB")
measured_value = 0.18

result = threshold.evaluate(measured_value)
print(f"Result: {result}")  # "PASS"
```

### Get All Thresholds for Multiple Contexts

```python
from spectrum.blue.thresholds import get_thresholds_for_context

# Comma-separated list of contexts
thresholds = get_thresholds_for_context("EU_AI_ACT,CFPB,EEOC")

print(f"Total thresholds: {len(thresholds)}")
for t in thresholds:
    print(f"  [{t.regulation}] {t.requirement}")
```

### Complete Assessment Example

```python
from spectrum.red.metrics import AdversarialMetrics
from spectrum.red.assessment import CompositeAssessment

# Create metrics from test results
metrics = AdversarialMetrics.from_fragility_score(
    fragility_score=0.22,
    attack_type="HopSkipJump",
    samples_tested=100,
    empirical_robustness_l2=0.15,
    empirical_robustness_linf=0.08
)

# Assess against regulatory context
assessment = CompositeAssessment.from_metrics(
    metrics=metrics,
    regulatory_context="EU_AI_ACT,CFPB"
)

print(f"Overall Robustness: {assessment.overall_robustness}")

for finding in assessment.findings:
    print(f"\n{finding.threshold.regulation}:")
    print(f"  Requirement: {finding.threshold.requirement}")
    print(f"  Status: {finding.status}")
    print(f"  Evidence: {finding.evidence}")
```

**Output:**
```
Overall Robustness: MARGINAL

EU_AI_ACT:
  Requirement: High-risk AI systems must demonstrate adequate robustness
  Status: PASS
  Evidence: attack_success_rate=0.220 vs threshold 0.25

CFPB:
  Requirement: Fair lending - model stability
  Status: MARGINAL
  Evidence: attack_success_rate=0.220 vs threshold 0.20
```

---

## Threshold Interpretation

### Status Values

- **PASS**: Metric meets the regulatory requirement
- **MARGINAL**: Metric is within 20% of threshold (warning zone)
- **FAIL**: Metric fails to meet the regulatory requirement

### Marginal Zone Logic

A "MARGINAL" status is assigned when:
- The metric passes the threshold, BUT
- The margin is less than 20% of the threshold value

**Example:**
```python
threshold = 0.25
measured = 0.22

# Passes threshold (0.22 < 0.25)
# Margin = |0.22 - 0.25| / 0.25 = 0.12 (12%)
# Since 12% < 20%, status is MARGINAL
```

---

## Metric Definitions

### attack_success_rate
**Description:** Fraction of samples successfully attacked (0.0-1.0)
**Lower is better**
**Used by:** EU_AI_ACT, CFPB, EEOC, NIST_AI_RMF, ISO_IEC_42001

### empirical_robustness_l2
**Description:** Mean L2 perturbation for successful attacks
**Lower is better**
**Used by:** EU_AI_ACT

### empirical_coverage
**Description:** Actual conformal prediction coverage on test set (0.0-1.0)
**Higher is better**
**Used by:** EU_AI_ACT, CFPB, NIST_AI_RMF

### max_psi
**Description:** Maximum Population Stability Index across features
**Lower is better**
**Used by:** CFPB

### disparate_impact_ratio
**Description:** Ratio of selection rates between groups (0.0-1.0)
**Higher is better (min 0.80 for four-fifths rule)**
**Used by:** EEOC

---

## Adding New Regulations

To add a new regulatory context:

1. **Define thresholds:**
```python
NEW_REGULATION_THRESHOLDS = [
    RegulatoryThreshold(
        regulation="NEW_REG",
        requirement="Description of requirement",
        metric_name="attack_success_rate",
        threshold_value=0.30,
        comparison="<",
        citation="Legal citation"
    ),
]
```

2. **Add to registry:**
```python
REGULATORY_THRESHOLDS["NEW_REG"] = NEW_REGULATION_THRESHOLDS
```

3. **Use immediately:**
```python
thresholds = get_thresholds_for_context("NEW_REG")
```

---

## CLI Integration

### Generate Report with Regulatory Context

```bash
# Initialize audit with regulatory context
spectrum lens init \
  --audit-name "Q1_Audit" \
  --client "Acme Corp" \
  --model ./model.pkl \
  --regulatory-context "EU_AI_ACT,CFPB"

# The regulatory context will be used to:
# 1. Select applicable thresholds
# 2. Assess compliance
# 3. Generate regulatory mapping in reports
```

---

## Best Practices

### 1. Always specify regulatory context
```python
# Good
assessment = CompositeAssessment.from_metrics(metrics, "EU_AI_ACT,CFPB")

# Bad (no context)
assessment = CompositeAssessment.from_metrics(metrics, "")
```

### 2. Use the most stringent thresholds
```python
# Automatically finds the most stringent threshold
threshold = get_threshold_by_metric("attack_success_rate", "EU_AI_ACT,CFPB,NIST_AI_RMF")
```

### 3. Document regulatory citations
```python
for finding in assessment.findings:
    print(f"Status: {finding.status}")
    print(f"Citation: {finding.threshold.citation}")
```

### 4. Monitor marginal zones
```python
marginal_findings = [f for f in assessment.findings if f.status == "MARGINAL"]
if marginal_findings:
    print("⚠ WARNING: Marginal compliance detected")
    for f in marginal_findings:
        print(f"  {f.threshold.regulation}: {f.evidence}")
```

---

## Troubleshooting

### Issue: "No thresholds found for context"
**Solution:** Check that the context ID is spelled correctly.
```python
# Correct
get_thresholds_for_context("EU_AI_ACT")

# Incorrect
get_thresholds_for_context("EU-AI-ACT")  # Wrong format
```

### Issue: "Metric not found in thresholds"
**Solution:** The metric may not have a defined threshold for that context.
```python
threshold = get_threshold_by_metric("unknown_metric", "EU_AI_ACT")
# Returns None if not found
```

### Issue: "Most stringent threshold is too strict"
**Solution:** This is by design - the system selects the most protective threshold across all specified contexts. Consider using a single context if you need less stringent requirements.

---

## Quick Reference Table

| Context | attack_success_rate | empirical_coverage | max_psi | disparate_impact |
|---------|---------------------|-------------------|---------|------------------|
| EU_AI_ACT | < 0.25 | >= 0.95 | - | - |
| CFPB | < 0.20 | >= 0.90 | < 0.25 | - |
| EEOC | < 0.30 | - | - | >= 0.80 |
| NIST_AI_RMF | < 0.15 | >= 0.95 | - | - |
| ISO_IEC_42001 | < 0.20 | - | - | - |

**Most Stringent:** `attack_success_rate < 0.15` (NIST_AI_RMF)

---

## Related Documentation

- **MODULE_UPDATES_SUMMARY.md** - Complete implementation details
- **CLI_GUIDE.md** - CLI usage examples
- **SPECS_GAP_ANALYSIS.md** - Feature comparison

---

**Last Updated:** December 3, 2025
**Version:** 0.1.0
