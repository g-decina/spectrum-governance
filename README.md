# Spectrum Governance

**AI/ML Governance Framework for Regulatory Compliance**

[![CI](https://github.com/YOUR_USERNAME/spectrum-governance/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/spectrum-governance/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

---

## Overview

Spectrum Governance is a hybrid Rust/Python framework for auditing AI/ML systems with a focus on regulatory compliance (EU AI Act, NIST AI RMF, CFPB). It provides:

- **Red Team**: Adversarial robustness testing (evasion, inference, extraction attacks)
- **Blue Team**: Defensive evaluation (uncertainty quantification, drift monitoring, explainability)
- **Lens**: Audit orchestration and lineage tracking

---

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/spectrum-governance.git
cd spectrum-governance

# Install with Poetry
poetry install

# With Blue Team dependencies (SHAP, MAPIE, Evidently)
poetry install --extras blue

# With Red Team dependencies (ART, ONNX)
poetry install --extras red
```

---

## Quick Start

### CLI Usage

```bash
# Run adversarial attack scan
spectrum red scan --model ./model.pkl --data ./test.csv --sample-size 100

# Generate SHAP explanations
spectrum blue explain --model ./model.pkl --data ./test.csv

# Check for data drift
spectrum blue drift --reference ./train.csv --current ./production.csv

# Initialize audit session
spectrum lens init --audit-name "Q1_Audit" --client "Acme Corp" --model ./model.pkl
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for complete CLI documentation.

### Python API

```python
from spectrum.blue.monitor import DriftCheck
from spectrum.blue.explain import SpectrumUncertaintyWrapper, generate_shap_explanations
from spectrum.red.attack import HopSkipJumpWrapper
import pandas as pd

# Drift monitoring
drift_results = DriftCheck(X_reference, X_current)
print(f"Max PSI: {drift_results['max_psi']}")
print(f"Drifted features: {drift_results['drifted_features']}")

# Uncertainty quantification
from spectrum.infra.types import RiskProfile, RiskLevel

risk_profile = RiskProfile(level=RiskLevel.HIGH, alpha=0.05)
wrapper = SpectrumUncertaintyWrapper(base_model=model, risk_profile=risk_profile)
wrapper.fit(X_calib, y_calib)
results = wrapper.predict(X_test)

# SHAP explanations
explanations = generate_shap_explanations(model, X_test, max_samples=10)

# Adversarial robustness
attack = HopSkipJumpWrapper(base_model=model)
metrics = attack.run(X_test)
print(f"Attack success rate: {metrics.attack_success_rate:.1%}")
```

---

## Architecture

```
spectrum-governance/
├── spectrum/               # Python package
│   ├── red/               # Adversarial attacks (ART wrappers)
│   ├── blue/              # Defense (MAPIE, SHAP, Evidently)
│   ├── lens/              # Audit logging & lineage
│   ├── cli/               # Typer CLI
│   └── infra/             # Core types, privacy, logging
├── spectrum-core/          # Rust workspace (performance-critical)
│   ├── spectrum-red/      # Rust attack implementations
│   ├── spectrum-blue/     # Rust defense implementations
│   └── spectrum-lens/     # Rust audit logging
└── tests/                 # Test suite
```

---

## Features

### Red Team - Adversarial Testing

| Attack Type | Description | Status |
|-------------|-------------|--------|
| HopSkipJump | Decision-based boundary attack | Implemented |
| ZOO | Zeroth-order optimization | Implemented |
| Boundary | Boundary attack | Implemented |
| Square | Query-efficient attack | Implemented |
| Membership Inference | Privacy attack | Implemented |
| Attribute Inference | Privacy attack | Implemented |
| Model Extraction | Surrogate model theft | Implemented |
| Data Poisoning | Backdoor injection | Implemented |

### Blue Team - Defensive Evaluation

| Feature | Description | Backend |
|---------|-------------|---------|
| Uncertainty Quantification | Conformal prediction intervals | MAPIE |
| Drift Monitoring | PSI-based distribution drift | Evidently |
| Explainability | Feature importance & contributions | SHAP |
| Model Hardening | Adversarial training wrapper | ART |

### Lens - Governance

| Feature | Description |
|---------|-------------|
| Audit Sessions | Initialize, track, close audit workflows |
| Lineage Tracking | OpenLineage integration with Marquez |
| RCIA Logging | Immutable audit event logging |

---

## Testing

```bash
# Run all Python tests
poetry run pytest tests/ -v

# Run specific test
poetry run pytest tests/test_e2e_credit_scoring.py -v

# Run Rust tests
cd spectrum-core && cargo test --release
```

---

## Requirements

- Python 3.10 - 3.12
- Rust 1.70+ (for spectrum-core)

### Core Dependencies

- pydantic >= 2.0
- numpy >= 1.26
- pandas >= 2.0
- scikit-learn >= 1.5
- typer >= 0.19

### Optional Dependencies

```bash
# Blue Team
poetry install --extras blue
# Includes: mapie, shap, evidently, openlineage-python

# Red Team
poetry install --extras red
# Includes: adversarial-robustness-toolbox, skl2onnx, onnxruntime
```

---

## Documentation

- [CLI Guide](CLI_GUIDE.md) - Complete CLI documentation
- [CLI Quick Reference](CLI_QUICK_REFERENCE.md) - Command cheat sheet
- [Quick Reference](QUICK_REFERENCE.md) - Python API reference
- [Lineage Guide](docs/LINEAGE.md) - OpenLineage + Marquez setup
- [Contributing](CONTRIBUTING.md) - Development guidelines
- [Security](SECURITY.md) - Security policy

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

```bash
# Development setup
poetry install --with dev
poetry run ruff check spectrum/ tests/
poetry run pytest
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [MAPIE](https://mapie.readthedocs.io/) - Conformal prediction
- [Evidently](https://evidentlyai.com/) - ML monitoring
- [SHAP](https://shap.readthedocs.io/) - Explainability
- [ART](https://adversarial-robustness-toolbox.org/) - Adversarial attacks
- [OpenLineage](https://openlineage.io/) - Data lineage
