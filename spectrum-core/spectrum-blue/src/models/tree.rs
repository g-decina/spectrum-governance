use super::{Model, TreeModel};

pub enum NodeType {
    Split { feature_index: usize, threshold: f64 },
    Leaf { value: f64 },
}

pub struct TreeNode {
    pub node_type: NodeType,
    pub left: Option<usize>,     // index of left child, None if leaf
    pub right: Option<usize>,    // index of right child, None if leaf
}

pub struct Tree {
    pub nodes: Vec<TreeNode>,
}

impl Tree {
    pub fn predict(&self, features: &[f64]) -> f64 {
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
    pub fn predict_subset(&self, features: &[f64], mask: &[bool]) -> f64 {
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
}

impl Model for Tree {
    fn predict(&self, features: &[f64]) -> f64 {
        self.predict(features)
    }
}

impl TreeModel for Tree {
    fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    fn root_index(&self) -> usize {
        0
    }
}