/// Naive SHAP implementation via exhaustive subset enumeration.
/// Runs in O(2^n) time — DO NOT use in production
/// Useful for testing and understanding SHAP properties

use crate::Tree;

pub struct NaiveSHAPExplainer;

impl NaiveSHAPExplainer {

    pub fn explain(&self, tree: &Tree, features: &[f64]) -> Vec<f64> {
        // Naive SHAP — runs in exponential time
        // Do not use in production
        let n = features.len();
        let mut contributions = vec![0.0; n];
        let num_subsets = 1 << n;

        for f in 0..n {
            let mut sum = 0.0;

            for bits in 0..num_subsets {
                let mask = self.int_to_mask(bits, n);

                if !mask[f] {
                    let v_without_f = tree.predict_subset(features, &mask);

                    let mut mask_with_f = mask.clone();
                    mask_with_f[f] = true;
                    let v_with_f = tree.predict_subset(features, &mask_with_f);

                    let weight = self.shapley_weight(mask.iter().filter(|&&b| b).count(), n);

                    sum += weight * (v_with_f - v_without_f);
                }
            }
            contributions[f] = sum;
        }

        contributions
    }

    fn shapley_weight(&self, s: usize, n: usize) -> f64 {
        // s = |S|, coalition size
        // n = total features
        // Weight = s! (n-s-1)! / n!    
        let mut weight = 1.0;
        for i in 1..=s {
            weight *= i as f64;
        }
        for i in 1..=(n - s - 1) {
            weight *= i as f64;
        }
        
        for i in 1..=n {
            weight /= i as f64;
        }

        weight
    }


    fn int_to_mask(&self, bits: usize, n: usize) -> Vec<bool> {
        (0..n).map(|i| (bits >> i) & 1 == 1).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TreeNode, NodeType};

    #[test]
    fn test_naive_shap_one_feature() {
        let tree = Tree { nodes: vec![
                TreeNode{
                    node_type: NodeType::Split{ feature_idx: 0, threshold: 0.1, 
                                                left_cover: 0.5, right_cover: 0.5 },
                    left: Some(1), right: Some(2),
                },
                TreeNode{
                    node_type: NodeType::Leaf { value: (1.0) },
                    left: None, right: None,
                },
                TreeNode{
                    node_type: NodeType::Leaf { value: (2.0) },
                    left: None, right: None,
                },
            ],
        };
        let explainer = NaiveSHAPExplainer;
        let shap = explainer.explain(&tree, &[0.05]);
        assert_eq!(shap, vec![-0.5])
    }

    #[test]
    fn test_naive_shap_three_features() {
        // 3-layer balanced tree using features 0, 1, 2
        //
        //                [0] f0 < 0.5
        //              /             \
        //        [1] f1<0.5        [2] f1<0.5
        //        /      \          /      \
        //    [3]f2<0.5 [4]f2    [5]f2    [6]f2<0.5
        //    /   \     /  \     /  \     /   \
        //  [7]  [8]  [9] [10] [11][12] [13] [14]
        //  v=0   1    2    3    4   5    6    7
        //
        let tree = Tree { nodes: vec![
            // [0] root: split on feature 0
            TreeNode {
                node_type: NodeType::Split { feature_idx: 0, threshold: 0.5,
                                            left_cover: 0.5, right_cover: 0.5 },
                left: Some(1), right: Some(2),
            },
            // [1] split on feature 1
            TreeNode {
                node_type: NodeType::Split { feature_idx: 1, threshold: 0.5,
                                            left_cover: 0.5, right_cover: 0.5 },
                left: Some(3), right: Some(4),
            },
            // [2] split on feature 1
            TreeNode {
                node_type: NodeType::Split { feature_idx: 1, threshold: 0.5,
                                            left_cover: 0.5, right_cover: 0.5 },
                left: Some(5), right: Some(6),
            },
            // [3] split on feature 2
            TreeNode {
                node_type: NodeType::Split { feature_idx: 2, threshold: 0.5,
                                            left_cover: 0.5, right_cover: 0.5 },
                left: Some(7), right: Some(8),
            },
            // [4] split on feature 2
            TreeNode {
                node_type: NodeType::Split { feature_idx: 2, threshold: 0.5,
                                            left_cover: 0.5, right_cover: 0.5 },
                left: Some(9), right: Some(10),
            },
            // [5] split on feature 2
            TreeNode {
                node_type: NodeType::Split { feature_idx: 2, threshold: 0.5,
                                            left_cover: 0.5, right_cover: 0.5 },
                left: Some(11), right: Some(12),
            },
            // [6] split on feature 2
            TreeNode {
                node_type: NodeType::Split { feature_idx: 2, threshold: 0.5,
                                            left_cover: 0.5, right_cover: 0.5 },
                left: Some(13), right: Some(14),
            },
            // Leaves [7]-[14] with values 0-7
            TreeNode { node_type: NodeType::Leaf { value: 0.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 1.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 2.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 3.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 4.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 5.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 6.0 }, left: None, right: None },
            TreeNode { node_type: NodeType::Leaf { value: 7.0 }, left: None, right: None },
        ]};

        let explainer = NaiveSHAPExplainer;

        // features = [0.25, 0.25, 0.25] → path: 0→1→3→7, prediction = 0.0
        // Base value (all features unknown) = average of all leaves = 3.5
        let shap = explainer.explain(&tree, &[0.25, 0.25, 0.25]);

        // SHAP values should sum to (prediction - base_value) = 0.0 - 3.5 = -3.5
        let sum: f64 = shap.iter().sum();
        assert!((sum - (-3.5)).abs() < 1e-10, "SHAP sum = {}, expected -3.5", sum);

        // Features higher in tree get more attribution:
        // - f0 at root: largest effect (selects left vs right half)
        // - f1 at depth 1: medium effect
        // - f2 at depth 2: smallest effect
        let expected = [-2.0, -1.0, -0.5];
        for (i, &v) in shap.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-10,
                    "SHAP[{}] = {}, expected {}", i, v, expected[i]);
        }
    }
}