# Test Fixtures

This directory contains reference data for equivalence testing against:
- **IBM ART** (Adversarial Robustness Toolbox) - for adversarial attacks
- **SHAP** - for explainability
- **MAPIE** - for conformal prediction

## Structure

```
fixtures/
├── art/           # ART reference outputs
├── shap/          # SHAP reference outputs
└── mapie/         # MAPIE reference outputs
```

## Generating Reference Data

Reference data should be generated using the Python reference implementations:

```bash
python tests/generate_art_reference.py --attack hopskipjump --output tests/fixtures/art/
```

## Equivalence Testing

All numerical algorithms must match reference implementations within ε = 10⁻⁵.
See CLAUDE.md for mathematical equivalence requirements.
