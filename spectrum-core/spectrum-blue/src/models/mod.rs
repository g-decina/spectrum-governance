// Base trait for a predictive model.
// KernelSHAP, fairness metrics, drift detection all use this.

pub mod tree;

pub use tree::{Tree, TreeNode, NodeType};

pub trait Model {
    fn predict(&self, features: &[f64]) -> f64;
}

// Trait for tree-based models that expose internal structure.
// TreeSHAP algorithm needs access to nodes for efficient computation.
pub trait TreeModel: Model {
    fn nodes(&self) -> &[TreeNode];
    fn root_idx(&self) -> usize;
}