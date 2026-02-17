//! Dynamic Knowledge Graph — GraphRAG for causal reasoning.
//!
//! Stores facts as directed edges between concept nodes. Supports:
//! - **BFS traversal** for cause/effect chain discovery
//! - **PageRank** for node importance scoring
//! - **Dijkstra** for shortest causal path between concepts
//!
//! Uses `petgraph::DiGraph` under the hood, with persistence via serde.

use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

// ─── Node & Edge Types ──────────────────────────────────────────────────────

/// Type of knowledge node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Entity,     // person, tool, object
    Concept,    // abstract idea
    Event,      // something that happened
    Preference, // user like/dislike
    Behavior,   // habitual pattern
}

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub label: String,
    pub node_type: NodeType,
    pub created_at: DateTime<Utc>,
    pub access_count: u32,
}

/// An edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    pub relation: String,
    pub strength: f32,
    pub evidence_count: u32,
    pub created_at: DateTime<Utc>,
}

/// A step in a causal chain.
#[derive(Debug, Clone, Serialize)]
pub struct CausalStep {
    pub from: String,
    pub relation: String,
    pub to: String,
    pub strength: f32,
}

/// Result of a knowledge query.
#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub query_node: String,
    pub chains: Vec<Vec<CausalStep>>,
    pub related_nodes: Vec<(String, f32)>, // (label, importance)
}

// ─── Knowledge Graph ────────────────────────────────────────────────────────

/// Thread-safe knowledge graph.
pub struct KnowledgeGraph {
    inner: RwLock<GraphInner>,
}

struct GraphInner {
    graph: DiGraph<KnowledgeNode, KnowledgeEdge>,
    label_to_idx: HashMap<String, NodeIndex>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(GraphInner {
                graph: DiGraph::new(),
                label_to_idx: HashMap::new(),
            }),
        }
    }

    /// Add or retrieve a node by label.
    pub fn ensure_node(&self, label: &str, node_type: NodeType) -> NodeIndex {
        let mut inner = self.inner.write().expect("kg lock poisoned");
        let lower = label.to_lowercase();
        if let Some(&idx) = inner.label_to_idx.get(&lower) {
            // Increment access count.
            if let Some(node) = inner.graph.node_weight_mut(idx) {
                node.access_count += 1;
            }
            return idx;
        }

        let node = KnowledgeNode {
            label: label.to_string(),
            node_type,
            created_at: Utc::now(),
            access_count: 1,
        };
        let idx = inner.graph.add_node(node);
        inner.label_to_idx.insert(lower, idx);
        idx
    }

    /// Add a directed fact: subject --[relation]--> object.
    /// If the edge already exists, strengthen it.
    pub fn add_fact(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
        subj_type: NodeType,
        obj_type: NodeType,
    ) {
        let from = self.ensure_node(subject, subj_type);
        let to = self.ensure_node(object, obj_type);

        let mut inner = self.inner.write().expect("kg lock poisoned");

        // Check if edge already exists.
        let existing = inner
            .graph
            .edges_connecting(from, to)
            .find(|e| e.weight().relation == relation);

        if let Some(edge_ref) = existing {
            let edge_id = edge_ref.id();
            if let Some(edge) = inner.graph.edge_weight_mut(edge_id) {
                edge.strength = (edge.strength + 0.1).min(1.0);
                edge.evidence_count += 1;
            }
        } else {
            inner.graph.add_edge(
                from,
                to,
                KnowledgeEdge {
                    relation: relation.to_string(),
                    strength: 0.5,
                    evidence_count: 1,
                    created_at: Utc::now(),
                },
            );
        }
    }

    /// BFS backward: find what causes/leads to this concept.
    pub fn query_causes(&self, label: &str, max_depth: usize) -> Vec<Vec<CausalStep>> {
        let inner = self.inner.read().expect("kg lock poisoned");
        let lower = label.to_lowercase();
        let Some(&start) = inner.label_to_idx.get(&lower) else {
            return vec![];
        };
        bfs_backward(&inner.graph, start, max_depth)
    }

    /// BFS forward: find effects from this concept.
    pub fn query_effects(&self, label: &str, max_depth: usize) -> Vec<Vec<CausalStep>> {
        let inner = self.inner.read().expect("kg lock poisoned");
        let lower = label.to_lowercase();
        let Some(&start) = inner.label_to_idx.get(&lower) else {
            return vec![];
        };
        bfs_forward(&inner.graph, start, max_depth)
    }

    /// Find the shortest path between two concepts.
    pub fn find_path(&self, from_label: &str, to_label: &str) -> Option<Vec<CausalStep>> {
        let inner = self.inner.read().expect("kg lock poisoned");
        let from_lower = from_label.to_lowercase();
        let to_lower = to_label.to_lowercase();
        let from = *inner.label_to_idx.get(&from_lower)?;
        let to = *inner.label_to_idx.get(&to_lower)?;

        // BFS shortest path.
        let mut visited: HashMap<NodeIndex, (NodeIndex, petgraph::graph::EdgeIndex)> =
            HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            if current == to {
                break;
            }
            for edge in inner.graph.edges(current) {
                let neighbor = edge.target();
                if neighbor != from && !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, (current, edge.id()));
                    queue.push_back(neighbor);
                }
            }
        }

        if !visited.contains_key(&to) && from != to {
            return None;
        }

        // Reconstruct path.
        let mut path = Vec::new();
        let mut current = to;
        while current != from {
            let (prev, edge_id) = visited.get(&current)?;
            let edge = inner.graph.edge_weight(*edge_id)?;
            let from_node = inner.graph.node_weight(*prev)?;
            let to_node = inner.graph.node_weight(current)?;
            path.push(CausalStep {
                from: from_node.label.clone(),
                relation: edge.relation.clone(),
                to: to_node.label.clone(),
                strength: edge.strength,
            });
            current = *prev;
        }
        path.reverse();
        Some(path)
    }

    /// Compute simple PageRank-like importance scores.
    pub fn pagerank(&self, iterations: usize, damping: f32) -> Vec<(String, f32)> {
        let inner = self.inner.read().expect("kg lock poisoned");
        let n = inner.graph.node_count();
        if n == 0 {
            return vec![];
        }

        let mut scores: HashMap<NodeIndex, f32> = inner
            .label_to_idx
            .values()
            .map(|&idx| (idx, 1.0 / n as f32))
            .collect();

        for _ in 0..iterations {
            let mut new_scores: HashMap<NodeIndex, f32> = HashMap::new();
            for (&idx, _) in &scores {
                new_scores.insert(idx, (1.0 - damping) / n as f32);
            }

            for (&idx, &score) in &scores {
                let out_edges: Vec<_> = inner.graph.edges(idx).collect();
                let out_degree = out_edges.len();
                if out_degree == 0 {
                    continue;
                }
                let share = damping * score / out_degree as f32;
                for edge in out_edges {
                    *new_scores.entry(edge.target()).or_insert(0.0) += share;
                }
            }

            scores = new_scores;
        }

        let mut result: Vec<(String, f32)> = scores
            .iter()
            .map(|(&idx, &score)| {
                let label = inner
                    .graph
                    .node_weight(idx)
                    .map(|n| n.label.clone())
                    .unwrap_or_default();
                (label, score)
            })
            .collect();

        result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Get graph statistics.
    pub fn stats(&self) -> (usize, usize) {
        let inner = self.inner.read().expect("kg lock poisoned");
        (inner.graph.node_count(), inner.graph.edge_count())
    }
}

// ─── BFS Helpers ─────────────────────────────────────────────────────────────

fn bfs_forward(
    graph: &DiGraph<KnowledgeNode, KnowledgeEdge>,
    start: NodeIndex,
    max_depth: usize,
) -> Vec<Vec<CausalStep>> {
    let mut chains = Vec::new();
    let mut queue: VecDeque<(NodeIndex, Vec<CausalStep>)> = VecDeque::new();
    queue.push_back((start, vec![]));

    while let Some((current, path)) = queue.pop_front() {
        if path.len() >= max_depth {
            if !path.is_empty() {
                chains.push(path);
            }
            continue;
        }

        let mut has_children = false;
        for edge in graph.edges(current) {
            has_children = true;
            let from_node = graph.node_weight(current).unwrap();
            let to_node = graph.node_weight(edge.target()).unwrap();
            let mut new_path = path.clone();
            new_path.push(CausalStep {
                from: from_node.label.clone(),
                relation: edge.weight().relation.clone(),
                to: to_node.label.clone(),
                strength: edge.weight().strength,
            });
            queue.push_back((edge.target(), new_path));
        }

        if !has_children && !path.is_empty() {
            chains.push(path);
        }
    }

    chains
}

fn bfs_backward(
    graph: &DiGraph<KnowledgeNode, KnowledgeEdge>,
    start: NodeIndex,
    max_depth: usize,
) -> Vec<Vec<CausalStep>> {
    // Traverse incoming edges.
    let mut chains = Vec::new();
    let mut queue: VecDeque<(NodeIndex, Vec<CausalStep>)> = VecDeque::new();
    queue.push_back((start, vec![]));

    while let Some((current, path)) = queue.pop_front() {
        if path.len() >= max_depth {
            if !path.is_empty() {
                chains.push(path);
            }
            continue;
        }

        let incoming: Vec<_> = graph
            .edges_directed(current, petgraph::Direction::Incoming)
            .collect();
        let mut has_parents = false;
        for edge in &incoming {
            has_parents = true;
            let from_node = graph.node_weight(edge.source()).unwrap();
            let to_node = graph.node_weight(current).unwrap();
            let mut new_path = path.clone();
            new_path.push(CausalStep {
                from: from_node.label.clone(),
                relation: edge.weight().relation.clone(),
                to: to_node.label.clone(),
                strength: edge.weight().strength,
            });
            queue.push_back((edge.source(), new_path));
        }

        if !has_parents && !path.is_empty() {
            chains.push(path);
        }
    }

    chains
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> KnowledgeGraph {
        let kg = KnowledgeGraph::new();
        kg.add_fact(
            "Cà phê",
            "GÂY_RA",
            "Mất ngủ",
            NodeType::Preference,
            NodeType::Event,
        );
        kg.add_fact(
            "Mất ngủ",
            "DẪN_ĐẾN",
            "Code bug",
            NodeType::Event,
            NodeType::Event,
        );
        kg.add_fact(
            "User",
            "THÍCH",
            "Cà phê",
            NodeType::Entity,
            NodeType::Preference,
        );
        kg.add_fact(
            "Code bug",
            "DẪN_ĐẾN",
            "Stress",
            NodeType::Event,
            NodeType::Concept,
        );
        kg
    }

    #[test]
    fn add_and_query_stats() {
        let kg = sample_graph();
        let (nodes, edges) = kg.stats();
        assert_eq!(nodes, 5); // User, Cà phê, Mất ngủ, Code bug, Stress
        assert_eq!(edges, 4);
    }

    #[test]
    fn forward_traversal_finds_effects() {
        let kg = sample_graph();
        let effects = kg.query_effects("cà phê", 3);
        assert!(!effects.is_empty(), "should find effects of coffee");
        // Should find: Cà phê -> Mất ngủ -> Code bug -> Stress
        let has_deep_chain = effects.iter().any(|chain| chain.len() >= 2);
        assert!(has_deep_chain, "should find multi-step chain");
    }

    #[test]
    fn backward_traversal_finds_causes() {
        let kg = sample_graph();
        let causes = kg.query_causes("code bug", 3);
        assert!(!causes.is_empty(), "should find causes of code bug");
        // Should trace back: Mất ngủ -> Cà phê
        let has_coffee = causes.iter().any(|chain| {
            chain
                .iter()
                .any(|step| step.from.contains("phê") || step.to.contains("phê"))
        });
        assert!(has_coffee, "should trace back to coffee");
    }

    #[test]
    fn find_path_works() {
        let kg = sample_graph();
        let path = kg.find_path("cà phê", "stress");
        assert!(path.is_some(), "should find path from coffee to stress");
        let path = path.unwrap();
        assert!(path.len() >= 2, "path should have at least 2 steps");
    }

    #[test]
    fn pagerank_identifies_important_nodes() {
        let kg = sample_graph();
        let ranks = kg.pagerank(20, 0.85);
        assert!(!ranks.is_empty());
        // "Code bug" and "Stress" are endpoint sinks, may rank differently.
        // Just verify it runs and returns meaningful values.
        let total: f32 = ranks.iter().map(|(_, s)| s).sum();
        assert!(
            (total - 1.0).abs() < 0.1,
            "pagerank should sum to ~1.0: {total}"
        );
    }

    #[test]
    fn duplicate_fact_strengthens_edge() {
        let kg = KnowledgeGraph::new();
        kg.add_fact("A", "R", "B", NodeType::Concept, NodeType::Concept);
        kg.add_fact("A", "R", "B", NodeType::Concept, NodeType::Concept);
        let (_, edges) = kg.stats();
        assert_eq!(edges, 1, "should not duplicate edge");
    }
}
