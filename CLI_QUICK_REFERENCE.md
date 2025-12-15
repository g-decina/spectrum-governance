# Spectrum CLI - Quick Reference Card

**Version:** 0.1.0  
**Print this page and keep at your workstation**

---

## Basic Commands

```bash
# Show help
spectrum --help
spectrum <command> --help

# Show version
spectrum --version
```

---

## Red Team - Adversarial Testing

### Run Attack Scan
```bash
spectrum red scan \
  --model <model.pkl> \
  --data <test.csv> \
  --sample-size 100
```

**Attack Success Rate:**
- `0-30%` = 🟢 GREEN (Robust)
- `31-60%` = 🟡 YELLOW (Moderate)
- `61-100%` = 🔴 RED (Vulnerable)

---

## Blue Team - Defensive Evaluation

### SHAP Explanations
```bash
spectrum blue explain \
  --model <model.pkl> \
  --data <test.csv> \
  --max-samples 10
```

### Drift Monitoring
```bash
spectrum blue drift \
  --reference <train.csv> \
  --current <production.csv>
```

**PSI Thresholds:**
- `< 0.10` = ✓ No drift
- `0.10-0.25` = ⚡ Monitor
- `> 0.25` = ⚠ Alert

### Uncertainty Quantification
```bash
spectrum blue uncertainty \
  --model <model.pkl> \
  --calibration-data <calib.csv> \
  --test-data <test.csv> \
  --risk-level HIGH
```

**Risk Levels:**
- `HIGH` = 95% confidence
- `MEDIUM` = 90% confidence
- `LOW` = 80% confidence

---

## Lens - Governance & Reporting

### Initialize Audit
```bash
spectrum lens init \
  --audit-name "Audit_Name" \
  --client "Client Corp" \
  --model <model.pkl> \
  --regulatory-context "EU_AI_ACT,CFPB"
```

### Check Status
```bash
spectrum lens status --audit-session <audit_dir>
```

### Generate Report
```bash
spectrum lens generate-report \
  --audit-session <audit_dir> \
  --format html
```

### Close Audit
```bash
spectrum lens close \
  --audit-session <audit_dir> \
  --archive
```

---

## Complete Workflow

```bash
# 1. Initialize
spectrum lens init --audit-name "Q1_Audit" --client "Corp" --model ./model.pkl

# 2. Test adversarial robustness
spectrum red scan --model ./model.pkl --data ./test.csv --sample-size 100

# 3. Explain predictions
spectrum blue explain --model ./model.pkl --data ./test.csv

# 4. Check drift
spectrum blue drift --reference ./train.csv --current ./prod.csv

# 5. Verify coverage
spectrum blue uncertainty --model ./model.pkl --calibration-data ./calib.csv --test-data ./test.csv --risk-level HIGH

# 6. Generate report
spectrum lens generate-report --audit-session ./audits/Q1_Audit/ --format html

# 7. Close & archive
spectrum lens close --audit-session ./audits/Q1_Audit/ --archive
```

---

## File Formats

**Models:** `.pkl`, `.joblib`  
**Data:** `.csv`, `.parquet`

---

## Common Options

- `--model, -m PATH` - Model file path
- `--data, -d PATH` - Data file path
- `--output, -o PATH` - Output directory
- `--verbose, -v` - Increase verbosity

---

## Regulatory Context Tags

- `EU_AI_ACT` - EU AI Act
- `CFPB` - Fair lending
- `EEOC` - Employment screening
- `HB_3773` - Illinois AI law
- `BIPA` - Biometric privacy

---

## Quick Tips

1. **Start small:** Use `--sample-size 50` first
2. **Use joblib:** Faster than pickle for large models
3. **Parquet format:** Much faster than CSV for large datasets
4. **Check help:** Every command has detailed `--help`

---

## Status Symbols

- ✓ Success / No issue
- ✗ Failure / Issue detected
- ⚠ Warning / Critical alert
- ⚡ Monitor / Moderate concern
- ℹ Information

---

## Error Troubleshooting

**"Model file not found"**  
→ Check path, use absolute paths

**"Failed to load model"**  
→ Verify .pkl or .joblib format

**HopSkipJump is slow**  
→ Reduce `--sample-size` and `--max-queries`

**"Template not found"**  
→ Expected, uses fallback HTML

---

**Documentation:** See `CLI_GUIDE.md`
**Version:** 0.1.0
