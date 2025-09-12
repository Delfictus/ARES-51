//! Decision tree implementation for ADP

use anyhow::Result;
use ndarray::Array1;

/// Decision tree for classification
pub struct DecisionTree {
    root: Option<Box<Node>>,
    max_depth: usize,
}

struct Node {
    feature_index: usize,
    threshold: f64,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    class_label: Option<String>,
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Result<Self> {
        Ok(Self {
            root: Some(Box::new(Self::build_sample_tree(max_depth)?)),
            max_depth,
        })
    }

    pub fn classify(&self, features: &Array1<f64>) -> Result<String> {
        if let Some(root) = &self.root {
            Ok(Self::classify_node(root, features))
        } else {
            Ok("unknown".to_string())
        }
    }

    fn classify_node(node: &Node, features: &Array1<f64>) -> String {
        if let Some(label) = &node.class_label {
            return label.clone();
        }

        if node.feature_index < features.len() {
            if features[node.feature_index] <= node.threshold {
                if let Some(left) = &node.left {
                    return Self::classify_node(left, features);
                }
            } else if let Some(right) = &node.right {
                return Self::classify_node(right, features);
            }
        }

        "unknown".to_string()
    }

    /// Build a sample decision tree
    fn build_sample_tree(max_depth: usize) -> Result<Node> {
        // Simple heuristic tree for packet routing decisions
        Ok(Node {
            feature_index: 1, // Priority
            threshold: 5.0,
            left: Some(Box::new(Node {
                feature_index: 2, // Packet size
                threshold: 1000.0,
                left: Some(Box::new(Node {
                    feature_index: 0,
                    threshold: 0.0,
                    left: None,
                    right: None,
                    class_label: Some("drop".to_string()),
                })),
                right: Some(Box::new(Node {
                    feature_index: 0,
                    threshold: 0.0,
                    left: None,
                    right: None,
                    class_label: Some("buffer".to_string()),
                })),
                class_label: None,
            })),
            right: Some(Box::new(Node {
                feature_index: 4, // Urgent flag
                threshold: 0.5,
                left: Some(Box::new(Node {
                    feature_index: 0,
                    threshold: 0.0,
                    left: None,
                    right: None,
                    class_label: Some("route".to_string()),
                })),
                right: Some(Box::new(Node {
                    feature_index: 0,
                    threshold: 0.0,
                    left: None,
                    right: None,
                    class_label: Some("modify".to_string()),
                })),
                class_label: None,
            })),
            class_label: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree() {
        let tree = DecisionTree::new(5).unwrap();
        let features = Array1::from_vec(vec![1.0, 7.0, 500.0, 0.0, 1.0]);
        let decision = tree.classify(&features).unwrap();
        assert!(!decision.is_empty());
    }
}
