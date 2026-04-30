/// spectrum-blue: Defense, explainability, and fairness analysis
///
/// This crate provides defensive capabilities:
/// - SHAP explainability (TreeExplainer)
/// - Fairness metrics and bias detection
/// - Drift detection and monitoring
/// - Conformal prediction (MAPIE integration)
///
/// Regulatory mapping:
/// - CFPB adverse action requirements (SHAP)
/// - ECOA, EEOC 4/5ths rule (fairness)

pub mod models;
pub mod explainability;

pub use models::{Model, NodeType, Tree, TreeModel, TreeNode};
pub use explainability::NaiveSHAPExplainer;