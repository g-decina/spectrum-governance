/// Adversarial attack implementations
///
/// This module contains pure Rust implementations of various adversarial attacks.
/// All attacks are designed for mathematical equivalence with IBM ART.
///
/// # Available Attacks
///
/// ## Decision-Based (Label-Only) - Classification
/// - **HopSkipJump**: Query-efficient boundary walking with gradient estimation
/// - **Boundary**: Random walk on decision boundary
///
/// ## Score-Based (Requires Probabilities) - Classification
/// - **ZOO**: Zeroth-order optimization with Adam
/// - **Square**: Random search with structured perturbations
///
/// ## Regression Attacks
/// - **OutputManipulation**: Discretize outputs into bins and attack like classifier
/// - **PredictionShift**: Maximize absolute change in predicted value
/// - **QuantileAttack**: Target conformal prediction intervals (break coverage)
///
/// # Usage
///
/// ```rust
/// use spectrum_red::attacks::{HopSkipJumpAttack, HopSkipJumpConfig};
///
/// let config = HopSkipJumpConfig::builder()
///     .max_iter(20)
///     .parallel(true)
///     .build();
///
/// let attack = HopSkipJumpAttack::new(config);
/// let metrics = attack.run(&model, &x_test)?;
/// ```

// Classification attacks
pub mod hop_skip_jump;
pub mod zoo;
pub mod boundary;
pub mod square;

// Regression attacks
pub mod regression_common;
pub mod output_manipulation;
pub mod prediction_shift;
pub mod quantile_attack;

// Re-export classification attack types
pub use hop_skip_jump::{HopSkipJumpAttack, HopSkipJumpConfig, HopSkipJumpError};
pub use zoo::{ZOOAttack, ZOOConfig, ZOOError};
pub use boundary::{BoundaryAttack, BoundaryConfig, BoundaryError};
pub use square::{SquareAttack, SquareConfig, SquareError};

// Re-export regression common types
pub use regression_common::{
    bootstrap_ci_mean, ConfidenceLevel, RegressionAttackError, RegressionMetrics, RegressionModel,
};

// Re-export regression attacks
pub use output_manipulation::{OutputManipulationAttack, OutputManipulationConfig};
pub use prediction_shift::{PredictionShiftAttack, PredictionShiftConfig};
pub use quantile_attack::{QuantileAttack, QuantileAttackConfig};
