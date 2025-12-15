# Contributing to Spectrum Governance

Thank you for your interest in contributing to Spectrum Governance! This document provides guidelines and instructions for contributing.

## Project Status

> **Note**: Spectrum Governance is currently in **pre-alpha** status. APIs may change significantly between versions. We welcome contributions, but please be aware that the project is evolving rapidly.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Check existing [GitHub Issues](https://github.com/YOUR_USERNAME/spectrum-governance/issues) to avoid duplicates
2. Use the latest version to see if the bug has been fixed

When submitting a bug report, include:

- Python version (`python --version`)
- Rust version if using spectrum-core (`rustc --version`)
- Operating system and version
- Minimal reproducible example
- Full error traceback
- Expected vs actual behavior

### Suggesting Features

Feature requests are welcome! Please:

1. Check existing issues for similar suggestions
2. Describe the use case and why existing features don't suffice
3. If possible, outline a proposed implementation

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**:
   ```bash
   pip install -e ".[blue,red]"
   pip install pytest ruff
   ```
3. **Make your changes** following the code style guidelines below
4. **Add tests** for any new functionality
5. **Run the test suite**:
   ```bash
   pytest tests/
   ```
6. **Run the linter**:
   ```bash
   ruff check spectrum/
   ```
7. **Update documentation** if needed
8. **Submit a pull request** with a clear description of changes

## Development Setup

### Python Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/spectrum-governance.git
cd spectrum-governance

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install with all dependencies
pip install -e ".[blue,red]"

# Install dev tools
pip install pytest ruff
```

### Rust Environment (for spectrum-core)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build Rust components
cd spectrum-core
cargo build --release

# Run Rust tests
cargo test
```

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/)
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Use type hints for function signatures
- Maximum line length: 100 characters

```python
# Good
def calculate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.1,
) -> DriftResult:
    """Calculate drift between reference and current data."""
    ...

# Avoid
def calculate_drift(reference, current, threshold=0.1):
    ...
```

### Rust

- Follow standard Rust conventions (`cargo fmt`)
- Use `cargo clippy` for linting
- Document public APIs with doc comments

```rust
/// Calculate adversarial perturbation for input sample.
///
/// # Arguments
/// * `x` - Input sample as 1D array
/// * `model` - Target model for attack
///
/// # Returns
/// Perturbed sample that causes misclassification
pub fn attack_single(&self, x: &Array1<f64>, model: &dyn Model) -> Result<Array1<f64>> {
    ...
}
```

## Testing

### Running Tests

```bash
# All Python tests
pytest tests/

# Specific test file
pytest tests/test_compliance_report.py -v

# End-to-end test
python tests/test_e2e_credit_scoring.py

# With coverage
pytest tests/ --cov=spectrum --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names that explain what is being tested

```python
def test_drift_check_detects_significant_shift():
    """DriftCheck should detect when feature distribution shifts significantly."""
    reference = pd.DataFrame({"feature": np.random.normal(0, 1, 1000)})
    current = pd.DataFrame({"feature": np.random.normal(2, 1, 1000)})  # Shifted

    result = DriftCheck(reference, current)

    assert result["drift_detected"] is True
    assert result["max_psi"] > 0.25
```

## Project Structure

```
spectrum-governance/
├── spectrum/               # Python package
│   ├── blue/              # Defense (uncertainty, drift, explainability)
│   ├── red/               # Attack (adversarial testing)
│   ├── lens/              # Governance (lineage, compliance)
│   ├── infra/             # Core types, logging
│   └── cli/               # Command-line interface
├── spectrum-core/         # Rust workspace
│   ├── spectrum-red/      # High-performance attacks
│   ├── spectrum-blue/     # High-performance defense
│   ├── spectrum-lens/     # Audit logging
│   └── spectrum-common/   # Shared types
├── tests/                 # Test suite
│   └── fixtures/          # Test data and ART reference files
└── docs/                  # Documentation
```

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add SHAP explanation caching for repeated predictions
fix: correct PSI calculation for categorical features
docs: add drift monitoring examples to README
test: add equivalence tests for HopSkipJump attack
refactor: extract threshold validation to separate module
```

## Questions?

- Open a [GitHub Issue](https://github.com/YOUR_USERNAME/spectrum-governance/issues) for bugs or features
- Start a [Discussion](https://github.com/YOUR_USERNAME/spectrum-governance/discussions) for questions

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
