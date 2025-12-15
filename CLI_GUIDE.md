# Spectrum CLI - Quick Start Guide

## Overview

The `spectrum` command-line interface provides access to ML purple teaming capabilities for adversarial auditing and regulatory compliance.

**Installation:**
```bash
pip install -e ".[blue,red_torch]"
```

After installation, the `spectrum` command will be available.

---

## Command Structure

```
spectrum [GLOBAL OPTIONS] COMMAND [ARGS]
```

**Global Options:**
- `--version, -v` - Show version and exit
- `--config PATH` - Path to configuration file (not yet implemented)
- `--verbose, -v` - Increase verbosity
- `--quiet, -q` - Suppress non-essential output

**Commands:**
- `red` - Adversarial testing
- `blue` - Defensive evaluation  
- `lens` - Governance & reporting

---

## Red Team - Adversarial Testing

### `spectrum red scan`

Execute adversarial attacks to assess model robustness.

**Usage:**
```bash
spectrum red scan \
  --model ./model.pkl \
  --data ./test.csv \
  --sample-size 100 \
  --output ./results/
```

**Options:**
- `--model, -m PATH` - Path to model file (.pkl, .joblib) **[required]**
- `--data, -d PATH` - Path to test data (.csv, .parquet) **[required]**
- `--sample-size, -n INT` - Number of samples to test [default: 100]
- `--max-queries INT` - Maximum model queries per sample [default: 10000]
- `--output, -o PATH` - Output directory for results
- `--verbose, -v` - Show detailed progress

**Example:**
```bash
# Run attack scan on 50 samples
spectrum red scan \
  --model ./credit_model.pkl \
  --data ./test_applications.csv \
  --sample-size 50
```

**Output:**
```
================================================================================
  SPECTRUM RED - ADVERSARIAL TESTING
================================================================================

✓ Model loaded: credit_model.pkl
✓ Data loaded: test_applications.csv (1000 rows, 15 columns)
ℹ Sampling 50 records from 1000 total
ℹ Test set: 50 samples, 14 features

================================================================================
  Running HopSkipJump Attack
================================================================================
ℹ This may take several minutes...

HopSkipJump: 100%|████████████| 50/50 [02:15<00:00,  2.70s/it]

✓ Attack complete

Attack Success Rate: 34.50%
Risk Level: YELLOW - Moderate
Interpretation: Vulnerability detected. Recommend hardening before deployment.

⚠ Model shows moderate vulnerability
```

**Interpreting Attack Success Rate:**
- **0-30% (GREEN)**: Strong robustness, acceptable for high-risk deployment
- **31-60% (YELLOW)**: Moderate vulnerability, recommend hardening
- **61-100% (RED)**: Highly vulnerable, not suitable for high-risk deployment

---

### `spectrum red llm-probe`

Execute LLM security probes (PLACEHOLDER - Not yet implemented).

**Usage:**
```bash
spectrum red llm-probe \
  --endpoint https://api.example.com/chat \
  --probes dan,promptinject
```

---

## Blue Team - Defensive Evaluation

### `spectrum blue explain`

Generate SHAP-based model explanations.

**Usage:**
```bash
spectrum blue explain \
  --model ./model.pkl \
  --data ./test.csv \
  --max-samples 10 \
  --output ./explanations/
```

**Options:**
- `--model, -m PATH` - Path to model file **[required]**
- `--data, -d PATH` - Path to data for explanation **[required]**
- `--max-samples, -n INT` - Number of samples to explain [default: 10]
- `--output, -o PATH` - Output directory

**Example:**
```bash
spectrum blue explain \
  --model ./credit_model.pkl \
  --data ./denied_applications.csv \
  --max-samples 20
```

**Output:**
```
================================================================================
  SPECTRUM BLUE - MODEL EXPLAINABILITY
================================================================================

✓ Model loaded: credit_model.pkl
✓ Data loaded: denied_applications.csv (150 rows, 15 columns)
ℹ Generating explanations for 20 samples...

✓ Explanations generated

Top Feature Contributions:
  1. credit_score: 0.4521
  2. debt_to_income_ratio: 0.3245
  3. num_late_payments: 0.2134
  4. employment_length_years: 0.1892
  5. annual_income: 0.1654
```

---

### `spectrum blue drift`

Analyze distribution drift between reference and current data.

**Usage:**
```bash
spectrum blue drift \
  --reference ./training_data.csv \
  --current ./production_data.csv \
  --psi-threshold 0.20 \
  --output ./drift_report/
```

**Options:**
- `--reference, -r PATH` - Path to reference data (e.g., training data) **[required]**
- `--current, -c PATH` - Path to current/production data **[required]**
- `--psi-threshold FLOAT` - PSI alert threshold [default: 0.20]
- `--output, -o PATH` - Output directory

**Example:**
```bash
spectrum blue drift \
  --reference ./train_2024.csv \
  --current ./production_2025_q1.csv
```

**Output:**
```
================================================================================
  SPECTRUM BLUE - DRIFT MONITORING
================================================================================

✓ Data loaded: train_2024.csv (10000 rows, 15 columns)
✓ Data loaded: production_2025_q1.csv (2500 rows, 15 columns)

✓ Drift analysis complete

Status: ⚡ Moderate drift detected - monitor closely
Max PSI: 0.1450
Alert Required: False
Drifted Features: 2

Per-Feature Drift Scores:
  ⚡ annual_income: 0.1450
  ⚡ credit_utilization: 0.1230
  ✓ credit_score: 0.0560
  ✓ debt_to_income_ratio: 0.0420
  ✓ num_late_payments: 0.0320
```

**PSI Thresholds:**
- **< 0.10**: No significant drift
- **0.10-0.25**: Moderate drift (monitor closely)
- **> 0.25**: Significant drift (immediate action required)

---

### `spectrum blue uncertainty`

Evaluate uncertainty quantification and coverage using conformal prediction.

**Usage:**
```bash
spectrum blue uncertainty \
  --model ./model.pkl \
  --calibration-data ./calib.csv \
  --test-data ./test.csv \
  --risk-level HIGH \
  --output ./uncertainty/
```

**Options:**
- `--model, -m PATH` - Path to model file **[required]**
- `--calibration-data, -c PATH` - Path to calibration data **[required]**
- `--test-data, -t PATH` - Path to test data **[required]**
- `--risk-level, -r STR` - Risk level: HIGH, MEDIUM, LOW [default: HIGH]
- `--output, -o PATH` - Output directory

**Risk Levels:**
- **HIGH**: 95% confidence (alpha=0.05)
- **MEDIUM**: 90% confidence (alpha=0.10)
- **LOW**: 80% confidence (alpha=0.20)

**Example:**
```bash
spectrum blue uncertainty \
  --model ./credit_model.pkl \
  --calibration-data ./calibration_set.csv \
  --test-data ./test_set.csv \
  --risk-level HIGH
```

**Output:**
```
================================================================================
  SPECTRUM BLUE - UNCERTAINTY QUANTIFICATION
================================================================================

✓ Model loaded: credit_model.pkl
✓ Data loaded: calibration_set.csv (500 rows, 15 columns)
✓ Data loaded: test_set.csv (200 rows, 15 columns)
ℹ Risk Profile: High
ℹ Required Confidence: 95.0%

ℹ Calibrating uncertainty wrapper...
✓ Calibration complete

ℹ Generating predictions with uncertainty...
✓ Predictions generated for 200 samples

Empirical Coverage: 96.50%
Required Coverage: 95.00%
✓ Coverage requirement MET
```

---

### `spectrum blue fairness`

Perform disparate impact analysis (PLACEHOLDER - Not yet implemented).

**Usage:**
```bash
spectrum blue fairness \
  --model ./model.pkl \
  --data ./predictions.csv \
  --demographics ./demographics.csv
```

---

## Lens - Governance & Reporting

### `spectrum lens init`

Initialize a new audit session.

**Usage:**
```bash
spectrum lens init \
  --audit-name "Q1_2025_Credit_Audit" \
  --client "Acme Financial" \
  --model ./credit_model.pkl \
  --regulatory-context "EU_AI_ACT,CFPB" \
  --output-dir ./audits/q1_2025/
```

**Options:**
- `--audit-name, -n STR` - Unique identifier for this audit **[required]**
- `--client, -c STR` - Client organization name **[required]**
- `--model, -m PATH` - Path to model being audited **[required]**
- `--regulatory-context, -r STR` - Comma-separated regulations
- `--output-dir, -o PATH` - Directory for audit artifacts

**Example:**
```bash
spectrum lens init \
  --audit-name "2025_Q1_Employment_Screening" \
  --client "BigCorp HR" \
  --model ./hr_screening_model.pkl \
  --regulatory-context "EEOC,HB_3773"
```

**Output:**
```
================================================================================
  SPECTRUM LENS - AUDIT INITIALIZATION
================================================================================

✓ Audit directory created: ./audits/q1_2025
✓ RCIA logger initialized: ./audits/q1_2025/logs/rcia_audit.jsonl
✓ Metadata saved: ./audits/q1_2025/audit_metadata.json

Audit Session Summary:
  Name: Q1_2025_Credit_Audit
  Client: Acme Financial
  Model: ./credit_model.pkl
  Regulations: EU_AI_ACT, CFPB
  Output: ./audits/q1_2025

✓ Audit session initialized successfully
```

---

### `spectrum lens status`

Display status of an audit session.

**Usage:**
```bash
spectrum lens status \
  --audit-session ./audits/q1_2025/ \
  --verbose
```

**Options:**
- `--audit-session, -a PATH` - Path to audit session directory **[required]**
- `--verbose, -v` - Show detailed event log

**Example:**
```bash
spectrum lens status --audit-session ./audits/q1_2025/
```

**Output:**
```
================================================================================
  SPECTRUM LENS - AUDIT STATUS
================================================================================

Audit Session: Q1_2025_Credit_Audit
  Client: Acme Financial
  Model: ./credit_model.pkl
  Created: 2025-01-15T09:00:00.000000
  Status: ACTIVE
  Regulations: EU_AI_ACT, CFPB

RCIA Events: 247 logged
```

---

### `spectrum lens generate-report`

Generate compliance report from audit session data.

**Usage:**
```bash
spectrum lens generate-report \
  --audit-session ./audits/q1_2025/ \
  --template tier1_forensic_audit \
  --format html \
  --output ./reports/q1_audit_report.html
```

**Options:**
- `--audit-session, -a PATH` - Path to audit session directory **[required]**
- `--template, -t STR` - Report template name [default: tier1_forensic_audit]
- `--format, -f STR` - Output format: html, pdf [default: html]
- `--output, -o PATH` - Output file path

**Available Templates:**
- `tier1_forensic_audit` - General purpose audit report
- `eu_ai_act_annex_iv` - EU AI Act Technical File (not yet implemented)
- `cfpb_model_validation` - CFPB lending compliance (not yet implemented)
- `eeoc_selection_audit` - Employment screening (not yet implemented)

**Example:**
```bash
spectrum lens generate-report \
  --audit-session ./audits/q1_2025/ \
  --template tier1_forensic_audit \
  --format html
```

---

### `spectrum lens close`

Close and finalize an audit session.

**Usage:**
```bash
spectrum lens close \
  --audit-session ./audits/q1_2025/ \
  --archive
```

**Options:**
- `--audit-session, -a PATH` - Path to audit session directory **[required]**
- `--archive` - Create tar.gz archive

**Example:**
```bash
spectrum lens close --audit-session ./audits/q1_2025/ --archive
```

**Output:**
```
================================================================================
  SPECTRUM LENS - AUDIT CLOSURE
================================================================================

✓ Audit session closed: Q1_2025_Credit_Audit
✓ Archive created: ./audits/q1_2025.tar.gz
```

---

## Complete Workflow Example

Here's a complete audit workflow:

```bash
# 1. Initialize audit session
spectrum lens init \
  --audit-name "Credit_Model_Audit_2025" \
  --client "Acme Financial" \
  --model ./models/credit_v2.pkl \
  --regulatory-context "EU_AI_ACT,CFPB" \
  --output-dir ./audits/credit_2025/

# 2. Run adversarial testing
spectrum red scan \
  --model ./models/credit_v2.pkl \
  --data ./data/test_set.csv \
  --sample-size 200 \
  --output ./audits/credit_2025/red_scan/

# 3. Generate explanations
spectrum blue explain \
  --model ./models/credit_v2.pkl \
  --data ./data/denied_apps.csv \
  --max-samples 50 \
  --output ./audits/credit_2025/explanations/

# 4. Check for drift
spectrum blue drift \
  --reference ./data/training_set.csv \
  --current ./data/production_q1.csv \
  --output ./audits/credit_2025/drift/

# 5. Evaluate uncertainty
spectrum blue uncertainty \
  --model ./models/credit_v2.pkl \
  --calibration-data ./data/calibration.csv \
  --test-data ./data/test_set.csv \
  --risk-level HIGH \
  --output ./audits/credit_2025/uncertainty/

# 6. Generate report
spectrum lens generate-report \
  --audit-session ./audits/credit_2025/ \
  --template tier1_forensic_audit \
  --format html \
  --output ./reports/credit_audit_2025.html

# 7. Close and archive
spectrum lens close \
  --audit-session ./audits/credit_2025/ \
  --archive
```

---

## Tips & Best Practices

### Model Formats
Supported model formats:
- `.pkl` / `.pickle` - Python pickle files
- `.joblib` - Joblib files (recommended for large arrays)

### Data Formats
Supported data formats:
- `.csv` - Comma-separated values
- `.parquet` / `.pq` - Parquet files (better for large datasets)

### Performance Tips
1. **Sample your data** - Use `--sample-size` to test on a subset first
2. **Start with small max-queries** - HopSkipJump can be slow; start with 5000 queries
3. **Use joblib for large models** - Faster loading than pickle
4. **Parquet for large datasets** - Much faster than CSV for 100k+ rows

### Regulatory Context Tags
Use these tags for `--regulatory-context`:
- `EU_AI_ACT` - EU AI Act compliance
- `CFPB` - Consumer Financial Protection Bureau (lending)
- `EEOC` - Equal Employment Opportunity Commission
- `HB_3773` - Illinois AI Employment Law
- `BIPA` - Illinois Biometric Information Privacy Act

---

## Troubleshooting

### "Model file not found"
Make sure you're using absolute paths or paths relative to your current directory.

### "Failed to load model"
- Check that the model file is not corrupted
- Verify you're using a supported format (.pkl, .joblib)
- Try loading the model in Python first to debug

### "Unsupported data format"
- Convert your data to CSV or Parquet
- Ensure the file extension matches the format

### HopSkipJump is very slow
- Reduce `--sample-size` (try 50 samples)
- Reduce `--max-queries` (try 5000)
- The attack is inherently slow; expect 1-3 seconds per sample

### "Template not yet implemented"
- Most regulatory templates are not yet built
- Use the default `tier1_forensic_audit` template
- Expect fallback HTML generation

---

## Current Implementation Status

**✅ Implemented:**
- `spectrum red scan` (HopSkipJump)
- `spectrum blue explain` (SHAP)
- `spectrum blue drift` (PSI)
- `spectrum blue uncertainty` (MAPIE conformal prediction)
- `spectrum lens init/status/generate-report/close`

**⚠️ Partial:**
- Report templates (only fallback HTML)
- LLM probes (placeholder)

**❌ Not Yet Implemented:**
- Feature constraints in attacks
- Attack campaigns
- ZooAttack and other adversarial methods
- `spectrum blue fairness` (DisparateImpactAnalyzer)
- EU AI Act Annex IV template
- CFPB/EEOC report templates
- PDF generation
- Configuration file support

---

## Getting Help

```bash
# Get help on any command
spectrum --help
spectrum red --help
spectrum blue explain --help

# Check version
spectrum --version
```

For more information, see:
- `README.md` - Project overview
- `QUICK_REFERENCE.md` - Python API reference
