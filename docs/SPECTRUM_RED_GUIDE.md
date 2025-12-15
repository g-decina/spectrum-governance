# Spectrum Red - Adversarial Attack Library

**High-performance adversarial robustness testing for ML models**

---

## Overview

Spectrum Red is a hybrid Rust/Python adversarial attack library designed for:

- **Regulatory compliance testing** (EU AI Act Article 15, NIST AI RMF)

- **Production ML auditing** (black-box attacks requiring only model predictions)

- **High-performance execution** (9-30x faster than pure Python)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API Layer                          │
│  HopSkipJumpWrapper, ZooAttackWrapper, BoundaryAttackWrapper│
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │    Backend Selection      │
            │    (auto/rust/art)        │
            └─────────────┬─────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼───────┐                 ┌─────────▼─────────┐
│  Rust Backend │                 │    ART Backend    │
│  (15x faster) │                 │    (fallback)     │
│               │                 │                   │
│ • Rayon       │                 │ • IBM ART         │
│ • ONNX        │                 │ • Pure Python     │
│ • Batched     │                 │                   │
└───────────────┘                 └───────────────────┘
```

---

## Installation

### Quick Install (Python only)

```bash
cd spectrum-governance
pip install -e ".[red_torch]"
```

### Full Install (with Rust backend - recommended)

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build Rust module
cd spectrum-core/spectrum-red
maturin develop --release --features python,onnx

# Install Python package
cd ../..
pip install -e ".[red_torch]"
```

### Verify Installation

```python
from spectrum.red.attack import RUST_AVAILABLE, ART_AVAILABLE
print(f"Rust backend: {'✓' if RUST_AVAILABLE else '✗'}")
print(f"ART backend:  {'✓' if ART_AVAILABLE else '✗'}")
```

---

## Quick Start

### Basic Attack

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from spectrum.red.attack import HopSkipJumpWrapper

# Train a model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier(n_estimators=100).fit(X[:800], y[:800])
X_test = X[800:810]  # 10 samples to attack

# Run HopSkipJump attack
attack = HopSkipJumpWrapper(
    base_model=model,
    backend="auto",      # Prefer Rust, fallback to ART
    max_iter=50,         # Boundary refinement iterations
    max_eval=1000,       # Query budget per iteration
    enable_onnx=True     # Convert model to ONNX for speed
)

metrics = attack.run(X_test)

# Results
print(f"Attack Success Rate: {metrics.attack_success_rate:.1%}")
print(f"Mean L2 Perturbation: {metrics.empirical_robustness_l2:.4f}")
print(f"Total Queries: {metrics.queries_used:,}")
```

### Interpreting Results

| Metric | Meaning | Good for Model |
|--------|---------|----------------|
| Attack Success Rate | % of samples successfully attacked | Lower is better |
| L2 Perturbation | Average perturbation magnitude | Higher is better |
| Queries Used | Computational cost | N/A |

**Regulatory Thresholds:**
- **HIGH risk models**: Success rate < 10%, L2 > 0.5
- **MEDIUM risk models**: Success rate < 30%, L2 > 0.3
- **LOW risk models**: Success rate < 50%

---

## Available Attacks

### HopSkipJump (Recommended)

**Decision-based black-box attack** - only needs hard labels.

```python
from spectrum.red.attack import HopSkipJumpWrapper

attack = HopSkipJumpWrapper(
    base_model=model,
    max_iter=64,         # More iterations = smaller perturbations
    max_eval=1000,       # Gradient estimation budget
    init_eval=25,        # Base gradient samples (scales with sqrt(iter))
    init_size=100,       # Initialization attempts
    backend="auto"
)
metrics = attack.run(X_test)
```

**Best for:** General robustness testing, models without probability outputs.

### ZOO (Zeroth-Order Optimization)

**Score-based attack** - uses probability outputs for finer optimization.

```python
from spectrum.red.attack import ZooAttackWrapper

attack = ZooAttackWrapper(
    base_model=model,
    max_iter=1000,
    learning_rate=0.01,
    binary_search_steps=10,
    backend="auto"
)
metrics = attack.run(X_test)
```

**Best for:** Models with probability outputs, targeted attacks.

### Boundary Attack

**Decision-based** - walks along the decision boundary.

```python
from spectrum.red.attack import BoundaryAttackWrapper

attack = BoundaryAttackWrapper(
    base_model=model,
    max_iter=5000,
    delta=0.01,           # Step size
    epsilon=0.01,         # Perturbation constraint
    backend="auto"
)
metrics = attack.run(X_test)
```

**Best for:** Fine-grained boundary analysis.

### Square Attack

**Score-based** - query-efficient random search.

```python
from spectrum.red.attack import SquareAttackWrapper

attack = SquareAttackWrapper(
    base_model=model,
    max_iter=1000,
    eps=0.3,              # L∞ constraint
    backend="auto"
)
metrics = attack.run(X_test)
```

**Best for:** L∞ bounded perturbations, query-limited scenarios.

---

## AdversarialMetrics

All attacks return an `AdversarialMetrics` object:

```python
@dataclass
class AdversarialMetrics:
    # Core robustness metrics
    attack_success_rate: float      # Fraction of successful attacks (0-1)
    samples_tested: int             # Total samples tested
    samples_successful: int         # Number of successful attacks

    # Perturbation magnitudes
    empirical_robustness_l2: float  # Mean L2 norm
    empirical_robustness_linf: float # Mean L∞ norm
    min_perturbation_l2: float      # Best-case (smallest) attack
    max_perturbation_l2: float      # Worst-case (largest) attack
    median_perturbation_l2: float   # Median attack

    # Query efficiency
    attack_type: str                # "HopSkipJump", "ZOO", etc.
    queries_used: int               # Total model queries
    avg_queries_per_sample: float   # Average queries per sample

    # Statistical measures
    confidence_interval_95: Tuple[float, float]  # Wilson CI for success rate
```

### Factory Methods

```python
# For evasion attacks (classification)
metrics = AdversarialMetrics.for_evasion_attack(
    attack_type="HopSkipJump",
    y_original=original_labels,
    y_adversarial=adversarial_labels,
    X_original=original_samples,
    X_adversarial=adversarial_samples,
    queries_used=10000
)

# For privacy inference attacks
metrics = AdversarialMetrics.for_inference_attack(
    attack_type="MembershipInference",
    privacy_leakage_score=0.65,
    samples_tested=1000
)
```

---

## Performance Optimization

### Backend Selection

```python
# Auto-select (recommended)
attack = HopSkipJumpWrapper(model, backend="auto")

# Force Rust (fastest, requires installation)
attack = HopSkipJumpWrapper(model, backend="rust")

# Force ART (fallback, always available)
attack = HopSkipJumpWrapper(model, backend="art")
```

### ONNX Conversion

ONNX conversion provides **2-15x speedup** by eliminating Python-Rust FFI overhead:

```python
# Enable ONNX (default: True)
attack = HopSkipJumpWrapper(model, enable_onnx=True)

# Disable if model doesn't support conversion
attack = HopSkipJumpWrapper(model, enable_onnx=False)
```

**Supported models:**
- `sklearn` estimators (via `skl2onnx`)
- PyTorch models (via `torch.onnx`)
- TensorFlow/Keras models (via `tf2onnx`)

### Parallelization

Rust backend uses Rayon for automatic multi-core execution:

```python
# Attack 100 samples in parallel
X_test = X[800:900]  # 100 samples
metrics = attack.run(X_test)  # Uses all CPU cores
```

---

## Benchmarks

Performance on 10 samples, 20 features, LogisticRegression:

| Backend | Time | Speedup | Success | L2 |
|---------|------|---------|---------|-----|
| ART (Python) | 0.25s | 1.0x | 100% | 8.05 |
| Rust + ONNX | 0.027s | **9.4x** | 100% | 8.47 |

Run benchmarks:

```bash
python tests/benchmark_attacks.py --config fast --test 10
```

---

## Integration with Spectrum Governance

### Wargame Runner

```python
from spectrum.red.wargame_runner import WargameRunner

runner = WargameRunner(
    model=model,
    X_test=X_test,
    risk_profile=RiskProfile(level=RiskLevel.HIGH)
)

results = runner.run_all_attacks()
compliance = runner.check_compliance(results)
```

### Compliance Reporting

```python
from spectrum.lens.compliance_report import ComplianceReport

report = ComplianceReport(
    model_name="CreditScorer",
    risk_level="HIGH",
    fragility_score=metrics.attack_success_rate,
    adversarial_metrics=metrics.to_dict()
)
```

---

## Rust Implementation Details

### HopSkipJump Algorithm

The Rust implementation (`spectrum-core/spectrum-red/src/attacks/hop_skip_jump.rs`) provides:

1. **Batched Gradient Estimation**
   - Single `model.predict()` call for N perturbations
   - 100-1000x faster than sequential calls

2. **Batched Initialization**
   - Process 64 candidates at once
   - Early exit on first adversarial found

3. **Optimized RNG**
   - Uses `rand_distr::StandardNormal` instead of Box-Muller
   - ~20% faster random number generation

4. **Pre-computed Labels**
   - Original labels computed once per sample
   - Saves 64 queries per sample (1 per iteration)

### Building from Source

```bash
cd spectrum-core/spectrum-red

# Development build (faster compilation)
maturin develop --features python,onnx

# Release build (optimized)
maturin develop --release --features python,onnx

# Run Rust tests
cargo test --features python,onnx
```

---

## Troubleshooting

### "ONNX conversion failed"

```
WARNING: ONNX conversion failed, using Python model (16x SLOWER)
```

**Fix:** Install `skl2onnx`:
```bash
pip install skl2onnx
```

### "Rust backend not available"

```
INFO: Rust attacks not installed, using ART backend
```

**Fix:** Build the Rust module:
```bash
cd spectrum-core/spectrum-red
maturin develop --release --features python,onnx
```

### Initialization failures

```
Error: Initialization failed: could not find adversarial sample after 1000 attempts
```

**Fix:** Increase `init_size`:
```python
attack = HopSkipJumpWrapper(model, init_size=5000)
```

---

## API Reference

### HopSkipJumpWrapper

```python
HopSkipJumpWrapper(
    base_model: BaseEstimator,      # sklearn/pytorch/tf model
    max_iter: int = 64,             # Boundary refinement iterations
    max_eval: int = 1000,           # Gradient estimation budget
    init_eval: int = 25,            # Base gradient samples
    init_size: int = 100,           # Initialization attempts
    backend: str = "auto",          # "auto", "rust", or "art"
    enable_onnx: bool = True        # Convert model to ONNX
)
```

### ZooAttackWrapper

```python
ZooAttackWrapper(
    base_model: BaseEstimator,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    binary_search_steps: int = 10,
    initial_const: float = 1e-3,
    backend: str = "auto",
    enable_onnx: bool = True
)
```

### BoundaryAttackWrapper

```python
BoundaryAttackWrapper(
    base_model: BaseEstimator,
    max_iter: int = 5000,
    delta: float = 0.01,
    epsilon: float = 0.01,
    step_adapt: float = 0.667,
    init_size: int = 100,
    backend: str = "auto",
    enable_onnx: bool = True
)
```

### SquareAttackWrapper

```python
SquareAttackWrapper(
    base_model: BaseEstimator,
    max_iter: int = 1000,
    eps: float = 0.3,
    p_init: float = 0.8,
    backend: str = "auto",
    enable_onnx: bool = True
)
```

---

## Mathematical Foundations

### Adversarial Examples - Formal Definition

Given a classifier `f: X → Y` and an input `x` with true label `y`, an **adversarial example** `x'` satisfies:

```
f(x') ≠ y  AND  ||x' - x||_p ≤ ε
```

Where:
- `||·||_p` is the Lp norm (typically L2 or L∞)
- `ε` is the perturbation budget

The goal of adversarial attacks is to find the **minimal perturbation**:

```
x* = argmin ||x' - x||_p  subject to  f(x') ≠ f(x)
        x'
```

---

### HopSkipJump Algorithm

**Reference:** Chen, Jordan & Wainwright (2019) - [arXiv:1904.02144](https://arxiv.org/pdf/1904.02144)

HopSkipJump is a **decision-based** attack requiring only hard labels (no probability scores).

#### Algorithm Overview

```
Input: Original sample x₀, Target model f
Output: Adversarial example x* with minimal ||x* - x₀||₂

1. INITIALIZE: Find any x_adv where f(x_adv) ≠ f(x₀)
2. For t = 1 to T:
   a. BOUNDARY SEARCH: x_boundary = BinarySearch(x₀, x_adv)
   b. GRADIENT ESTIMATE: g ≈ ∇d(x_boundary)
   c. BOUNDARY WALK: x_adv = Project(x_boundary - η·g)
3. Return x_adv
```

#### Step 1: Initialization

Find any adversarial sample by random search:

```python
for i in 1..N:
    x_rand = sample_uniform(0, 1, d)  # Random d-dimensional vector
    if f(x_rand) ≠ f(x₀):
        return x_rand
```

**Complexity:** O(N) queries, where N is typically 100-1000.

#### Step 2a: Binary Search for Boundary

Given `x₀` (original) and `x_adv` (adversarial), find the decision boundary:

```
low = 0, high = 1
while (high - low) > δ:
    mid = (low + high) / 2
    x_mid = (1 - mid) · x₀ + mid · x_adv    # Linear interpolation
    if f(x_mid) ≠ f(x₀):
        high = mid    # Move toward original
    else:
        low = mid     # Move toward adversarial
return x_boundary = (1 - high) · x₀ + high · x_adv
```

**Complexity:** O(log(1/δ)) queries per search.

#### Step 2b: Gradient Estimation (Monte Carlo)

The gradient of the distance function cannot be computed directly (black-box model). We estimate it using **finite differences**:

```
g = 0
for i in 1..B:
    u_i ~ N(0, I)              # Sample random direction
    u_i = u_i / ||u_i||₂       # Normalize to unit sphere
    x_+ = x_boundary + σ·u_i   # Perturb in direction u_i

    if f(x_+) ≠ f(x₀):
        g = g + u_i            # Direction points toward adversarial
    else:
        g = g - u_i            # Direction points toward original

return g / ||g||₂              # Normalize gradient estimate
```

**Mathematical Justification:**

The boundary normal `n` at point `x_boundary` satisfies:

```
n = lim    E[u · sign(f(x_boundary + σu) ≠ f(x₀))]
    σ→0
```

This converges to the true gradient with rate O(1/√B) where B is the number of samples.

**Complexity:** O(B) queries per iteration.

#### Step 2c: Boundary Walk

Update adversarial example by stepping toward original:

```
step_size = γ · ||x_adv - x₀||₂ / √(t + 1)
x_candidate = x_adv - step_size · g
x_adv = BinarySearch(x₀, x_candidate)  # Project back to boundary
```

**Step Size Schedule:**

The `1/√(t+1)` decay ensures convergence while allowing rapid initial progress:
- Early iterations: Large steps (coarse refinement)
- Late iterations: Small steps (fine refinement)

#### Convergence Analysis

**Theorem (Chen et al., 2019):** Under mild assumptions on the decision boundary smoothness, HopSkipJump converges to a local minimum of the perturbation function with rate:

```
||x_t - x*||₂ = O(1/√t)
```

**Query Complexity:** Total queries = O(T · B · log(1/δ))

For typical parameters (T=64, B=100, δ=10⁻⁵):
- ~64 × 100 × 17 ≈ 100,000 queries per sample

---

### ZOO Attack (Zeroth-Order Optimization)

**Reference:** Chen et al. (2017) - [arXiv:1708.03999](https://arxiv.org/abs/1708.03999)

ZOO uses **coordinate-wise gradient estimation** with probability scores.

#### Optimization Objective

Minimize the C&W loss function:

```
L(x') = ||x' - x||₂² + c · max(Z(x')_y - max{Z(x')_i : i ≠ y}, -κ)
```

Where:
- `Z(x')` = logits (pre-softmax scores)
- `y` = original class
- `c` = regularization constant (found via binary search)
- `κ` = confidence margin

#### Coordinate-Wise Gradient Estimation

For each coordinate `i`:

```
∂L/∂x_i ≈ (f(x + h·e_i) - f(x - h·e_i)) / (2h)
```

Where `e_i` is the i-th standard basis vector and `h` is a small step (typically 10⁻⁴).

**Complexity:** 2d queries per gradient (d = number of features).

#### Optimization

Use Adam optimizer with estimated gradients:

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
x_{t+1} = x_t - α · m_t / (√v_t + ε)
```

---

### Boundary Attack

**Reference:** Brendel, Rauber & Bethge (2017) - [arXiv:1712.04248](https://arxiv.org/abs/1712.04248)

#### Algorithm

Starting from an adversarial point, walk along the decision boundary toward the original:

```
1. Initialize x_adv = random adversarial sample
2. For t = 1 to T:
   a. Sample orthogonal perturbation: δ ⊥ (x_adv - x₀)
   b. Move toward original: x' = x_adv + ε_orth·δ - ε_step·(x_adv - x₀)/||x_adv - x₀||
   c. If f(x') ≠ f(x₀): x_adv = x'
3. Return x_adv
```

#### Orthogonal Perturbation

To stay on the boundary while reducing distance:

```
δ = random_normal(d)
δ = δ - (δ · v) · v    # Remove component parallel to v = (x_adv - x₀)
δ = δ / ||δ||₂         # Normalize
```

---

### Square Attack

**Reference:** Andriushchenko et al. (2019) - [arXiv:1912.00049](https://arxiv.org/abs/1912.00049)

#### L∞ Attack via Random Search

Square Attack uses **localized square-shaped perturbations**:

```
1. Initialize: δ = 0, p = initial_probability
2. For t = 1 to T:
   a. Sample square location (i, j) and size s
   b. Propose perturbation: δ' = δ; δ'[i:i+s, j:j+s] = ±ε
   c. If f(x + δ') ≠ f(x) AND margin improves:
      δ = δ'
   d. Decay p geometrically
3. Return x + δ
```

#### Query Efficiency

The square structure exploits image locality - a successful square perturbation in one region suggests nearby regions may also be vulnerable.

**Complexity:** O(T) queries (typically T = 1000-5000).

---

### Perturbation Metrics

#### L2 Norm

```
||x' - x||₂ = √(Σᵢ (x'ᵢ - xᵢ)²)
```

**Interpretation:** Euclidean distance - measures total perturbation magnitude.

#### L∞ Norm

```
||x' - x||∞ = maxᵢ |x'ᵢ - xᵢ|
```

**Interpretation:** Maximum change to any single feature.

#### Empirical Robustness

Given a set of successful attacks `{(x_i, x'_i)}`:

```
ρ_emp = (1/N) Σᵢ ||x'ᵢ - xᵢ||₂
```

**Interpretation:** Average perturbation needed to fool the model. Higher = more robust.

---

### Statistical Confidence

#### Wilson Score Confidence Interval

For attack success rate `p = k/n` (k successes in n trials):

```
CI_95 = (p + z²/2n ± z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)
```

Where `z = 1.96` for 95% confidence.

This is more accurate than normal approximation for small samples.

---

### Regulatory Thresholds

Based on EU AI Act Article 15 and NIST AI RMF guidance:

| Risk Level | Attack Success | L2 Threshold | Interpretation |
|------------|----------------|--------------|----------------|
| HIGH | < 10% | > 0.5 | Credit, healthcare, justice |
| MEDIUM | < 30% | > 0.3 | Marketing, recommendations |
| LOW | < 50% | N/A | Non-critical applications |

**Rationale:**
- HIGH risk models must resist 90%+ of attacks with meaningful perturbations
- The L2 threshold ensures attacks aren't trivially small (imperceptible changes)

---

## References

- [HopSkipJump Paper](https://arxiv.org/pdf/1904.02144) - Chen et al., 2019
- [ZOO Attack Paper](https://arxiv.org/abs/1708.03999) - Chen et al., 2017
- [Boundary Attack Paper](https://arxiv.org/abs/1712.04248) - Brendel et al., 2017
- [Square Attack Paper](https://arxiv.org/abs/1912.00049) - Andriushchenko et al., 2019
- [IBM ART](https://adversarial-robustness-toolbox.org/) - Reference implementation
- [EU AI Act Article 15](https://artificialintelligenceact.eu/article/15/) - Robustness requirements

---

**Last Updated:** December 2024
