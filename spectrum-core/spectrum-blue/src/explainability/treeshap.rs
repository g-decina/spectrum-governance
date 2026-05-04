//! TreeSHAP: Efficient O(TLD²) algorithm for exact SHAP values on trees.
//! Based on Lundberg et al. "Consistent Individualized Feature Attribution for Tree Ensembles"
//!
//! Implements the polynomial-time algorithm that computes exact Shapley values
//! by tracking path information during tree traversal.

use crate::{Tree, NodeType};

/// Path element tracking a feature encountered during tree traversal.
#[derive(Clone, Copy, Debug)]
struct PathElement {
    feature_idx: i64,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

pub struct TreeSHAPExplainer;

impl TreeSHAPExplainer {
    pub fn explain(&self, tree: &Tree, features: &[f64]) -> Vec<f64> {
        let n_features = features.len();
        let mut shap_values = vec![0.0; n_features];
        let max_path = tree.nodes.len() + 2;
        let mut path = vec![PathElement {
            feature_idx: -1,
            zero_fraction: 1.0,
            one_fraction: 1.0,
            pweight: 0.0,
        }; max_path];

        self.tree_shap_recursive(tree, features, 0, &mut path, 0, 1.0, 1.0, -1, &mut shap_values);
        shap_values
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_shap_recursive(
        &self,
        tree: &Tree,
        x: &[f64],
        node_idx: usize,
        path: &mut [PathElement],
        unique_depth: usize,  // Current depth (index of next slot to fill)
        zero_frac: f64,
        one_frac: f64,
        feature_idx: i64,
        shap: &mut [f64],
    ) {
        // Extend path with this feature (always extends, even for initial -1)
        self.extend_path(path, unique_depth, zero_frac, one_frac, feature_idx);
        let new_depth = unique_depth + 1;

        let node = &tree.nodes[node_idx];

        match &node.node_type {
            NodeType::Leaf { value } => {
                // At leaf: compute SHAP contributions for each feature in path
                // Start at 1 to skip the dummy element at index 0
                for i in 1..new_depth {
                    let w = self.unwound_path_sum(path, new_depth, i);
                    let contrib = w * (path[i].one_fraction - path[i].zero_fraction) * value;
                    if path[i].feature_idx >= 0 {
                        shap[path[i].feature_idx as usize] += contrib;
                    }
                }
            }

            NodeType::Split { feature_idx: split_feat, threshold, left_cover, right_cover } => {
                let feat = *split_feat as i64;
                let total = left_cover + right_cover;
                let hot_zero_frac = left_cover / total;
                let cold_zero_frac = right_cover / total;

                let go_left = x[*split_feat] < *threshold;
                let (hot_idx, cold_idx, hw, cw) = if go_left {
                    (node.left.unwrap(), node.right.unwrap(), hot_zero_frac, cold_zero_frac)
                } else {
                    (node.right.unwrap(), node.left.unwrap(), cold_zero_frac, hot_zero_frac)
                };

                // Check if we've already split on this feature
                let path_idx = self.find_feature_index(path, new_depth, feat);

                let (incoming_zero, incoming_one, recurse_depth) = if let Some(idx) = path_idx {
                    // Feature already in path - unwind it and merge fractions
                    let iz = path[idx].zero_fraction;
                    let io = path[idx].one_fraction;
                    self.unwind_path(path, new_depth, idx);
                    (iz, io, new_depth - 1)
                } else {
                    // New feature
                    (1.0, 1.0, new_depth)
                };

                // Save path state before recursion
                let state: Vec<PathElement> = path[..new_depth].to_vec();

                // Hot path recursion
                self.tree_shap_recursive(
                    tree, x, hot_idx, path, recurse_depth,
                    incoming_zero * hw, incoming_one, feat, shap
                );

                // Restore path before cold recursion
                path[..new_depth].copy_from_slice(&state);

                // Cold path recursion
                self.tree_shap_recursive(
                    tree, x, cold_idx, path, recurse_depth,
                    incoming_zero * cw, 0.0, feat, shap
                );
            }
        }
    }

    /// Add a feature to the path and update weights.
    fn extend_path(&self, path: &mut [PathElement], unique_depth: usize, zf: f64, of: f64, feat: i64) {
        path[unique_depth] = PathElement {
            feature_idx: feat,
            zero_fraction: zf,
            one_fraction: of,
            pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
        };

        // Update weights using the recurrence from Lundberg et al.
        for i in (0..unique_depth).rev() {
            let m = (unique_depth + 1) as f64;
            let fi = (i + 1) as f64;
            path[i + 1].pweight += of * path[i].pweight * fi / m;
            path[i].pweight = zf * path[i].pweight * (unique_depth as f64 - i as f64) / m;
        }
    }

    /// Remove feature at path_idx from the path and update weights.
    fn unwind_path(&self, path: &mut [PathElement], unique_depth: usize, path_idx: usize) {
        let of = path[path_idx].one_fraction;
        let zf = path[path_idx].zero_fraction;
        let mut next_one_portion = path[unique_depth - 1].pweight;

        for i in (0..unique_depth - 1).rev() {
            if of != 0.0 {
                let tmp = path[i].pweight;
                let m = unique_depth as f64;
                path[i].pweight = next_one_portion * m / (((i + 1) as f64) * of);
                next_one_portion = tmp - path[i].pweight * zf * (unique_depth as f64 - 1.0 - i as f64) / m;
            } else {
                let m = unique_depth as f64;
                path[i].pweight = path[i].pweight * m / (zf * (unique_depth as f64 - 1.0 - i as f64));
            }
        }

        // Shift elements after path_idx to fill the gap
        for i in path_idx..unique_depth - 1 {
            path[i].feature_idx = path[i + 1].feature_idx;
            path[i].zero_fraction = path[i + 1].zero_fraction;
            path[i].one_fraction = path[i + 1].one_fraction;
        }
    }

    fn find_feature_index(&self, path: &[PathElement], depth: usize, feat: i64) -> Option<usize> {
        (0..depth).find(|&i| path[i].feature_idx == feat)
    }

    /// Compute the sum of Shapley weights for unwinding feature at path_idx.
    fn unwound_path_sum(&self, path: &[PathElement], unique_depth: usize, path_idx: usize) -> f64 {
        let of = path[path_idx].one_fraction;
        let zf = path[path_idx].zero_fraction;

        if of == 0.0 && zf == 0.0 {
            return 0.0;
        }

        let mut next_one_portion = path[unique_depth - 1].pweight;
        let mut total = 0.0;

        if of != 0.0 {
            for i in (0..unique_depth - 1).rev() {
                let tmp = next_one_portion / (((i + 1) as f64) * of);
                total += tmp;
                next_one_portion = path[i].pweight - tmp * zf * ((unique_depth - 1 - i) as f64);
            }
        } else {
            for i in (0..unique_depth - 1).rev() {
                total += path[i].pweight / (zf * ((unique_depth - 1 - i) as f64));
            }
        }

        total * (unique_depth as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TreeNode, NodeType};
    use crate::explainability::NaiveSHAPExplainer;

    fn two_feature_tree() -> Tree {
        // f0 at root, f1 at both children
        //        [0] f0 < 0.5
        //       /           \
        //   [1] f1<0.5    [2] f1<0.5
        //   /    \        /    \
        // [3]   [4]     [5]   [6]
        // v=0   v=1     v=2   v=3
        Tree { nodes: vec![
            TreeNode { node_type: NodeType::Split { feature_idx: 0, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(1), right: Some(2) },
            TreeNode { node_type: NodeType::Split { feature_idx: 1, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(3), right: Some(4) },
            TreeNode { node_type: NodeType::Split { feature_idx: 1, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(5), right: Some(6) },
            TreeNode { node_type: NodeType::Leaf { value: 0.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 1.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 2.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 3.0 }, left: None, right: None },
        ]}
    }

    #[test]
    fn test_treeshap_two_features() {
        let tree = two_feature_tree();
        let x = [0.25, 0.25];  // Only 2 features
        let ts = TreeSHAPExplainer.explain(&tree, &x);
        let ns = NaiveSHAPExplainer.explain(&tree, &x);
        eprintln!("TreeSHAP: {:?}", ts);
        eprintln!("NaiveSHAP: {:?}", ns);
        // Base = (0+1+2+3)/4 = 1.5, pred = 0.0, diff = -1.5
        // f0 goes left (to leaves 0,1), f1 goes left (to leaf 0)
        let ts_sum: f64 = ts.iter().sum();
        let ns_sum: f64 = ns.iter().sum();
        assert!((ts_sum - ns_sum).abs() < 1e-10, "TreeSHAP sum={}, NaiveSHAP sum={}", ts_sum, ns_sum);
        for i in 0..2 {
            assert!((ts[i] - ns[i]).abs() < 1e-10, "Feature {}: TreeSHAP={}, NaiveSHAP={}", i, ts[i], ns[i]);
        }
    }

    fn one_feature_tree() -> Tree {
        Tree { nodes: vec![
            TreeNode {
                node_type: NodeType::Split { feature_idx: 0, threshold: 0.1, left_cover: 0.5, right_cover: 0.5 },
                left: Some(1), right: Some(2),
            },
            TreeNode { node_type: NodeType::Leaf { value: 1.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 2.0 }, left: None, right: None },
        ]}
    }

    fn three_feature_tree() -> Tree {
        Tree { nodes: vec![
            TreeNode { node_type: NodeType::Split { feature_idx: 0, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(1), right: Some(2) },
            TreeNode { node_type: NodeType::Split { feature_idx: 1, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(3), right: Some(4) },
            TreeNode { node_type: NodeType::Split { feature_idx: 1, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(5), right: Some(6) },
            TreeNode { node_type: NodeType::Split { feature_idx: 2, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(7), right: Some(8) },
            TreeNode { node_type: NodeType::Split { feature_idx: 2, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(9), right: Some(10) },
            TreeNode { node_type: NodeType::Split { feature_idx: 2, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(11), right: Some(12) },
            TreeNode { node_type: NodeType::Split { feature_idx: 2, threshold: 0.5, left_cover: 0.5, right_cover: 0.5 }, left: Some(13), right: Some(14) },
            TreeNode { node_type: NodeType::Leaf { value: 0.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 1.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 2.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 3.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 4.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 5.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 6.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 7.0 }, left: None, right: None },
        ]}
    }

    #[test]
    fn test_treeshap_one_feature() {
        let tree = one_feature_tree();
        let ts = TreeSHAPExplainer.explain(&tree, &[0.05]);
        let ns = NaiveSHAPExplainer.explain(&tree, &[0.05]);
        assert!((ts[0] - ns[0]).abs() < 1e-10, "TreeSHAP={}, NaiveSHAP={}", ts[0], ns[0]);
    }

    #[test]
    fn test_treeshap_three_features() {
        let tree = three_feature_tree();
        let x = [0.25, 0.25, 0.25];
        let ts = TreeSHAPExplainer.explain(&tree, &x);
        let ns = NaiveSHAPExplainer.explain(&tree, &x);
        for i in 0..3 {
            assert!((ts[i] - ns[i]).abs() < 1e-10, "Feature {}: TreeSHAP={}, NaiveSHAP={}", i, ts[i], ns[i]);
        }
    }

    #[test]
    fn test_treeshap_efficiency() {
        let tree = three_feature_tree();
        let shap = TreeSHAPExplainer.explain(&tree, &[0.25, 0.25, 0.25]);
        let sum: f64 = shap.iter().sum();
        assert!((sum - (-3.5)).abs() < 1e-10, "SHAP sum = {}, expected -3.5", sum);
    }
}
