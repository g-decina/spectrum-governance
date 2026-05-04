/// Naive SHAP implementation via exhaustive subset enumeration.
/// Runs in O(2^n) time — DO NOT use in production
/// Useful for testing and understanding SHAP properties

use crate::{Tree, TreeNode, NodeType};

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
            let mut count = 0;

            for bits in 0..num_subsets {
                let mask = self.int_to_mask(bits, n);
                
                if !mask[f] {
                    let v_without_f = tree.predict_subset(features, &mask);

                    let mut mask_with_f = mask.clone();
                    mask_with_f[f] = true;
                    let v_with_f = tree.predict_subset(features, &mask_with_f);

                    sum += v_with_f - v_without_f;
                    count += 1;
                }
            }
            contributions[f] = sum / count as f64;
        }

        contributions
    }


    fn int_to_mask(&self, bits: usize, n: usize) -> Vec<bool> {
        (0..n).map(|i| (bits >> i) & 1 == 1).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_shap() {
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
}