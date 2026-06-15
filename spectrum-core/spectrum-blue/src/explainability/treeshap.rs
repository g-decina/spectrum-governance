//! # TreeSHAP: Polynomial-Time Exact Shapley Values for Decision Trees
//!
//! Computes exact SHAP values for tree-based models in O(LD²) time, where L is the
//! number of leaves and D is the maximum tree depth. This is exponentially faster
//! than naive O(2ⁿ) enumeration.
//!
//! ## Reference
//!
//! Lundberg, S. M., Erion, G. G., & Lee, S. I. (2018).
//! "Consistent Individualized Feature Attribution for Tree Ensembles."
//! arXiv:1802.03888
//!
//! ## Algorithm
//!
//! TreeSHAP exploits tree structure to avoid explicit coalition enumeration:
//!
//! 1. Only features on the root-to-leaf path affect the prediction
//! 2. Coalitions are tracked implicitly via `pweight` (accumulated Shapley weights)
//! 3. Hot/cold path recursion explores both "feature present" and "feature absent" cases
//!
//! The key data structure is `PathElement`, which tracks:
//! - `zero_fraction`: weight when feature is absent from coalition
//! - `one_fraction`: weight when feature is present in coalition
//! - `pweight`: accumulated Shapley weight via Lundberg recurrence relations

use crate::{NodeType, Tree};

/// Tracks coalition state for one feature during tree traversal.
///
/// - `feature_idx`: Feature index (-1 for dummy initial element)
/// - `zero_fraction`: Fraction of data taking "other branch" (feature unknown)
/// - `one_fraction`: Fraction taking "our branch" (feature known)
/// - `pweight`: Accumulated Shapley weight encoding |S|!(n-|S|-1)!/n! sums
#[derive(Default, Clone, Copy, Debug)]
struct PathElement {
    feature_idx: i64,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

/// Computes exact Shapley values for decision trees in polynomial time.
///
/// # Example
///
/// ```ignore
/// let explainer = TreeSHAPExplainer;
/// let shap_values = explainer.explain(&tree, &features);
/// assert!((shap_values.iter().sum::<f64>() - (prediction - base_value)).abs() < 1e-10);
/// ```
pub struct TreeSHAPExplainer;

impl TreeSHAPExplainer {
    /// Compute SHAP values for all features.
    ///
    /// # Returns
    /// Vector of SHAP values where `sum(shap) = prediction - base_value`.
    pub fn explain(&self, tree: &Tree, features: &[f64]) -> Vec<f64> {
        let n_features = features.len();
        let mut shap = vec![0.0; n_features];

        let max_path_length = tree.nodes.len() + 2;
        let mut path = vec![
            PathElement {
                feature_idx: -1,
                zero_fraction: 0.0,
                one_fraction: 0.0,
                pweight: 0.0
            }; max_path_length
        ];

        path[0] = PathElement {
            feature_idx: -1,
            zero_fraction: 1.0,
            one_fraction: 1.0,
            pweight: 0.0
        };

        self.tree_shap_recursive(
            tree, features, 0, &mut path, 0, 1.0, 1.0, -1, &mut shap
        );

        shap
    }

    /// Recursive traversal computing SHAP contributions at each leaf.
    ///
    /// At splits: recurses into both hot (feature present) and cold (feature absent) branches.
    /// At leaves: computes weighted contributions via `unwound_path_sum`.
    #[allow(clippy::too_many_arguments)]
    fn tree_shap_recursive(
        &self,
        tree: &Tree,
        x: &[f64],
        node_idx: usize,
        path: &mut [PathElement],
        unique_depth: usize,
        zero_frac: f64,
        one_frac: f64,
        feature_idx: i64,
        shap: &mut [f64],
    ) {
        self.extend_path(path, unique_depth, zero_frac, one_frac, feature_idx);
        let new_depth = unique_depth + 1;
        let node = &tree.nodes[node_idx];

        match node.node_type {
            NodeType::Leaf { value } => {
                for i in 1..new_depth {
                    let w = self.unwound_path_sum(path, new_depth, i);
                    let contrib = w * (path[i].one_fraction - path[i].zero_fraction) * value;
                    let feat = path[i].feature_idx;
                    if feat >= 0 {
                        shap[feat as usize] += contrib;
                    }
                }
            }
            NodeType::Split { feature_idx: split_feat, threshold, left_cover, right_cover } => {
                let total_cover = left_cover + right_cover;
                let go_left = x[split_feat] < threshold;

                let (hot_idx, cold_idx, hot_zero_frac, cold_zero_frac) = if go_left {
                    (node.left.unwrap(), node.right.unwrap(),
                    left_cover / total_cover, right_cover / total_cover)
                } else {
                    (node.right.unwrap(), node.left.unwrap(),
                    right_cover / total_cover, left_cover / total_cover)
                };

                // Handle repeated features by unwinding previous occurrence
                let (incoming_zero, incoming_one, recurse_depth) =
                    match self.find_feature_index(path, new_depth, split_feat as i64) {
                        Some(idx) => {
                            let iz = path[idx].zero_fraction;
                            let io = path[idx].one_fraction;
                            self.unwind_path(path, unique_depth, idx);
                            (iz, io, new_depth - 1)
                        }
                        None => (1.0, 1.0, new_depth)
                };

                let saved: Vec<PathElement> = path[0..new_depth].to_vec();

                // Hot path: feature is present (one_frac preserved)
                self.tree_shap_recursive(
                    tree, x, hot_idx, path, recurse_depth,
                    incoming_zero * hot_zero_frac, incoming_one, split_feat as i64, shap
                );

                path[0..new_depth].copy_from_slice(&saved);

                // Cold path: feature is absent (one_frac = 0)
                self.tree_shap_recursive(
                    tree, x, cold_idx, path, recurse_depth,
                    incoming_zero * cold_zero_frac, 0.0, split_feat as i64, shap
                );
            }
        }
    }

    /// Extend path with new feature, updating pweights via Lundberg recurrence.
    ///
    /// Recurrence (iterating backwards to avoid overwriting):
    /// - `pweight[i+1] += of * pweight[i] * (i+1) / m`
    /// - `pweight[i] = zf * pweight[i] * (unique_depth - i) / m`
    fn extend_path(
        &self,
        path: &mut [PathElement],
        unique_depth: usize,
        zf: f64,
        of: f64,
        feat: i64,
    ) {
        let m = (unique_depth + 1) as f64;

        path[unique_depth] = PathElement {
            feature_idx: feat,
            zero_fraction: zf,
            one_fraction: of,
            pweight: if unique_depth == 0 { 1.0 } else { 0.0 }
        };

        if unique_depth == 0 {
            path[unique_depth].pweight = 1.0
        } else {
            for i in (0..unique_depth).rev() {
                path[i+1].pweight += of * path[i].pweight * ((i + 1) as f64) / m;
                path[i].pweight = zf * path[i].pweight * ((unique_depth - i) as f64) / m
            }
        }
    }

    /// Remove a feature from path, restoring pweights to pre-extension state.
    ///
    /// Used when a feature appears multiple times on the root-to-leaf path.
    fn unwind_path(
        &self,
        path: &mut [PathElement],
        unique_depth: usize,
        path_idx: usize,
    ) {
        let of = path[path_idx].one_fraction;
        let zf = path[path_idx].zero_fraction;
        let mut next_one_portion = path[unique_depth-1].pweight;
        let m = unique_depth as f64;

        for i in (0..unique_depth - 1).rev() {
            if of != 0.0 {
                let tmp = path[i].pweight;
                path[i].pweight = next_one_portion * m / (((i + 1) as f64) * of);
                next_one_portion = tmp - path[i].pweight * zf * (m-1.0-i as f64) / m
            } else {
                path[i].pweight = path[i].pweight * m / (zf * (m-1.0-i as f64))
            }
        }

        for i in path_idx..(unique_depth - 1) {
            path[i].feature_idx = path[i + 1].feature_idx;
            path[i].zero_fraction = path[i + 1].zero_fraction;
            path[i].one_fraction = path[i + 1].one_fraction;
        }
    }

    /// Compute total Shapley weight for feature at `path_idx`.
    ///
    /// "Virtually unwinds" the feature to sum weights across all coalition sizes.
    fn unwound_path_sum(
        &self,
        path: &[PathElement],
        unique_depth: usize,
        path_idx: usize,
    ) -> f64 {
        let of = path[path_idx].one_fraction;
        let zf = path[path_idx].zero_fraction;

        if of == 0.0 && zf == 0.0 {
            return 0.0;
        }

        let mut next_one_portion = path[unique_depth - 1].pweight;
        let mut total = 0.0;

        if of != 0.0 {
            for i in (0..unique_depth - 1).rev() {
                let tmp = next_one_portion / ((i + 1) as f64 * of);
                total += tmp;
                next_one_portion = path[i].pweight - tmp * zf * (unique_depth - 1 - i) as f64;
            }
        } else {
            for i in (0..unique_depth - 1).rev() {
                total += path[i].pweight / (zf * (unique_depth - 1 - i) as f64);
            }
        }

        total * unique_depth as f64
    }

    /// Find feature in path buffer. Returns `Some(index)` if found.
    fn find_feature_index(
        &self,
        path: &[PathElement],
        depth: usize,
        feat: i64,
    ) -> Option<usize> {
        (0..depth).find(|&i| path[i].feature_idx == feat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TreeNode, NodeType};
    use crate::explainability::NaiveSHAPExplainer;

    fn one_feature_tree() -> Tree {
        Tree {
            nodes: vec![
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 0,
                        threshold: 0.1,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(1),
                    right: Some(2),
                },
                TreeNode {
                    node_type: NodeType::Leaf { value: 1.0 },
                    left: None,
                    right: None,
                },
                TreeNode {
                    node_type: NodeType::Leaf { value: 2.0 },
                    left: None,
                    right: None,
                },
            ],
        }
    }

    fn two_feature_tree() -> Tree {
        Tree {
            nodes: vec![
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 0,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(1),
                    right: Some(2),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 1,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(3),
                    right: Some(4),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 1,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(5),
                    right: Some(6),
                },
                TreeNode { node_type: NodeType::Leaf { value: 0.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 1.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 2.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 3.0 }, left: None, right: None },
            ],
        }
    }

    fn three_feature_tree() -> Tree {
        Tree {
            nodes: vec![
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 0,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(1),
                    right: Some(2),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 1,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(3),
                    right: Some(4),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 1,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(5),
                    right: Some(6),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 2,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(7),
                    right: Some(8),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 2,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(9),
                    right: Some(10),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 2,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(11),
                    right: Some(12),
                },
                TreeNode {
                    node_type: NodeType::Split {
                        feature_idx: 2,
                        threshold: 0.5,
                        left_cover: 0.5,
                        right_cover: 0.5,
                    },
                    left: Some(13),
                    right: Some(14),
                },
                TreeNode { node_type: NodeType::Leaf { value: 0.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 1.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 2.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 3.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 4.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 5.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 6.0 }, left: None, right: None },
                TreeNode { node_type: NodeType::Leaf { value: 7.0 }, left: None, right: None },
            ],
        }
    }

    #[test]
    fn test_treeshap_one_feature() {
        let tree = one_feature_tree();
        let x = [0.05];

        let ts = TreeSHAPExplainer.explain(&tree, &x);
        let ns = NaiveSHAPExplainer.explain(&tree, &x);

        assert!(
            (ts[0] - ns[0]).abs() < 1e-10,
            "TreeSHAP={}, NaiveSHAP={}",
            ts[0],
            ns[0]
        );
    }

    #[test]
    fn test_treeshap_two_features() {
        let tree = two_feature_tree();
        let x = [0.25, 0.25];

        let ts = TreeSHAPExplainer.explain(&tree, &x);
        let ns = NaiveSHAPExplainer.explain(&tree, &x);

        let ts_sum: f64 = ts.iter().sum();
        let ns_sum: f64 = ns.iter().sum();
        assert!(
            (ts_sum - ns_sum).abs() < 1e-10,
            "Sum mismatch: TreeSHAP={}, NaiveSHAP={}",
            ts_sum,
            ns_sum
        );

        for i in 0..2 {
            assert!(
                (ts[i] - ns[i]).abs() < 1e-10,
                "Feature {}: TreeSHAP={}, NaiveSHAP={}",
                i,
                ts[i],
                ns[i]
            );
        }
    }

    #[test]
    fn test_treeshap_three_features() {
        let tree = three_feature_tree();
        let x = [0.25, 0.25, 0.25];

        let ts = TreeSHAPExplainer.explain(&tree, &x);
        let ns = NaiveSHAPExplainer.explain(&tree, &x);

        for i in 0..3 {
            assert!(
                (ts[i] - ns[i]).abs() < 1e-10,
                "Feature {}: TreeSHAP={}, NaiveSHAP={}",
                i,
                ts[i],
                ns[i]
            );
        }
    }

    #[test]
    fn test_treeshap_efficiency() {
        let tree = three_feature_tree();
        let x = [0.25, 0.25, 0.25];

        let shap = TreeSHAPExplainer.explain(&tree, &x);
        let sum: f64 = shap.iter().sum();

        assert!(
            (sum - (-3.5)).abs() < 1e-10,
            "SHAP sum = {}, expected -3.5",
            sum
        );
    }
}