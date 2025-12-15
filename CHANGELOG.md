# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with Python package and Rust workspace
- Blue Team module: uncertainty quantification with MAPIE, drift detection with Evidently, SHAP explanations
- Red Team module: HopSkipJump attack wrapper for adversarial testing
- Lens module: OpenLineage integration, compliance reporting, DOCX report generation
- CLI interface with `spectrum` command
- End-to-end credit scoring example
- Regulatory threshold configurations (CFPB, NIST AI RMF)

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - Unreleased

Initial pre-alpha release.

### Features
- **spectrum.blue**: Defense mechanisms
  - `SpectrumUncertaintyWrapper`: Conformal prediction with coverage guarantees
  - `DriftCheck`: PSI-based drift detection
  - `generate_shap_explanations`: SHAP-based feature explanations

- **spectrum.red**: Adversarial testing
  - `HopSkipJumpWrapper`: Black-box adversarial attack

- **spectrum.lens**: Governance
  - `LineageTracker`: OpenLineage event emission
  - `ComplianceReport`: Pydantic-validated audit reports
  - `ReportBuilder`: DOCX report generation

- **spectrum.infra**: Core infrastructure
  - `RiskProfile`, `RiskLevel`: Risk-based governance configuration
  - Logging and privacy utilities

### Known Limitations
- Red Team module is ~40% complete
- No fairness/bias detection yet
- Rust backend (spectrum-core) is experimental

---

[Unreleased]: https://github.com/YOUR_USERNAME/spectrum-governance/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/YOUR_USERNAME/spectrum-governance/releases/tag/v0.1.0
