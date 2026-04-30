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

pub mod placeholder;

pub use placeholder::*;

enum NodeType {
    Split { feature_index: usize, threshold: f64 },
    Leaf { value: f64 },
}

struct TreeNode {
    node_type: NodeType,
    left: Option<usize>,    // index of left child, None if leaf
    right: Option<usize>,    // index of right child, None if leaf
}

struct Tree {
    nodes: Vec<TreeNode>,
}

impl Tree {
    fn predict(&self, features: &[f64]) -> f64 {
        let mut current_idx = 0;
        loop {
            let node = &self.nodes[current_idx];
            match &node.node_type {
                NodeType::Leaf { value } => { return *value }
                NodeType::Split { feature_index, threshold } => {
                    if features[*feature_index] < *threshold {
                        current_idx = node.left.unwrap()
                    } else { current_idx = node.right.unwrap() }
                }
            }
        }
    }


    // Predict using only the features in `mask`.
    // mask[i] = true enables features i, false means "average both branches"
    fn predict_subset(&self, features: &[f64], mask: &[bool]) -> f64 {
        self.predict_node(0, features, mask)
    }


    fn predict_node(&self, node_idx: usize, features: &[f64], mask: &[bool]) -> f64 {
        let node = &self.nodes[node_idx];

        match &node.node_type {
            NodeType::Leaf { value } => { 
                // Base case: return leaf value
                *value 
            }

            NodeType::Split { feature_index, threshold } => {
                if mask[*feature_index] {
                    // Feature is KNOWN: go left or right as normal
                    if features[*feature_index] < *threshold {
                        self.predict_node(node.left.unwrap(), features, mask)
                    } else { 
                        self.predict_node(node.right.unwrap(), features, mask)
                    }
                } else {
                    // Feature is UNKNOWN: average over both branches
                    let left_val = self.predict_node(node.left.unwrap(), features, mask);
                    let right_val = self.predict_node(node.right.unwrap(), features, mask);
                    (left_val + right_val) / 2.0
                }
            }
        }
    }


    fn int_to_mask(&self, bits: usize, n: usize) -> Vec<bool> {
        (0..n).map(|i| (bits >> i) & 1 == 1).collect()
    }


    fn _naive_shap(&self, features: &[f64]) -> Vec<f64> {
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
                    let v_without_f = self.predict_subset(features, &mask);

                    let mut mask_with_f = mask.clone();
                    mask_with_f[f] = true;
                    let v_with_f = self.predict_subset(features, &mask_with_f);

                    sum += v_with_f - v_without_f;
                    count += 1;
                }
            }
            contributions[f] = sum / count as f64;
        }

        contributions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tree() {
        let tree = Tree {
            nodes: vec![
                TreeNode{
                    node_type: NodeType::Split{ feature_index: 0, threshold: 0.1 },
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
        
        // Test: feature[0] = 0.05 → smaller than 0.1 → goes left → expect 1.0
        assert_eq!(tree.predict(&[0.05]), 1.0);

        // Test: feature[0] = 0.5 → greater than 0.1 → goes right → expect 2.0
        assert_eq!(tree.predict(&[0.5]), 2.0);
    }

    #[test]
    fn test_predict_subset() {
        let tree = Tree {
            nodes: vec![
                TreeNode{
                    node_type: NodeType::Split{ feature_index: 0, threshold: 0.1 },
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

        assert_eq!(tree.predict_subset(&[0.05], &[true]), 1.0);
        assert_eq!(tree.predict_subset(&[0.05], &[false]), 1.5);
    }

    #[test]
    fn test_naive_shap() {
        let tree = Tree {nodes: vec![
                TreeNode{
                    node_type: NodeType::Split{ feature_index: 0, threshold: 0.1 },
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
        let shap = tree._naive_shap(&[0.05]);
        assert_eq!(shap, vec![-0.5])
    }
}