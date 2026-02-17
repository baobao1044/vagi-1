//! System 2 Reasoning — Draft-level MCTS (TreeSearchAgent).
//!
//! Unlike the token-level MCTS in `crate::mcts`, this operates on **complete drafts**.
//! Each node in the tree holds an entire response candidate. Expansion generates
//! multiple full drafts via the model, and each is evaluated end-to-end by the
//! WASI verifier and world model. Branches with low scores are pruned — this is
//! how hallucinations die before reaching the user.
//!
//! # Architecture
//!
//! ```text
//!                     [Root: prompt]
//!                    /      |       \
//!              [Draft A] [Draft B] [Draft C]     ← Expand: model.infer() × 3
//!              score=0.8  score=0.3  score=0.7
//!                 |        PRUNED      |
//!              [A.1] [A.2]          [C.1] [C.2]  ← Re-expand best branches
//!              0.85  0.72           0.90  0.65
//! ```
//!
//! After N iterations, the highest-scoring leaf is returned.

use std::time::Instant;

use anyhow::Result;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::model_runtime::ModelRuntime;
use crate::models::{ThinkingEvent, VerifierRequest};
use crate::verifier::Verifier;
use crate::world_model::WorldModel;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Tunable parameters for System 2 reasoning.
#[derive(Debug, Clone)]
pub struct ThinkConfig {
    /// Number of SELECT→EXPAND→EVAL→BACKPROP iterations.
    pub iterations: usize,
    /// Number of candidate drafts generated per expansion.
    pub candidates_per_expand: usize,
    /// UCT exploration constant.
    pub exploration_c: f32,
    /// Branches with avg score below this are pruned.
    pub prune_threshold: f32,
    /// Maximum tree depth (refinement levels).
    pub max_depth: u32,
    /// Max tokens per draft generation.
    pub max_tokens_per_draft: usize,
}

impl Default for ThinkConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            candidates_per_expand: 3,
            exploration_c: 1.414,
            prune_threshold: 0.15,
            max_depth: 4,
            max_tokens_per_draft: 128,
        }
    }
}

// ─── Thought Node ────────────────────────────────────────────────────────────

/// A node in the Thought Tree. Each holds a **complete draft** (not a token).
#[derive(Debug, Clone)]
struct ThoughtNode {
    /// The full draft text at this node (empty for root).
    state: String,
    /// Indices of child nodes in the arena.
    children: Vec<usize>,
    /// Number of times this node has been visited/evaluated.
    visits: u32,
    /// Sum of all scores received through this node.
    total_value: f64,
    /// Parent index (None for root).
    parent: Option<usize>,
    /// Whether the WASI verifier passed this draft.
    verifier_pass: bool,
    /// Risk score from the world model [0, 1].
    risk_score: f32,
    /// Tree depth (0 = root).
    depth: u32,
    /// Whether this node has already been expanded.
    expanded: bool,
    /// Whether this branch was pruned.
    pruned: bool,
}

impl ThoughtNode {
    fn root(prompt: String) -> Self {
        Self {
            state: prompt,
            children: Vec::new(),
            visits: 0,
            total_value: 0.0,
            parent: None,
            verifier_pass: false,
            risk_score: 0.0,
            depth: 0,
            expanded: false,
            pruned: false,
        }
    }

    fn child(state: String, parent_idx: usize, depth: u32) -> Self {
        Self {
            state,
            children: Vec::new(),
            visits: 0,
            total_value: 0.0,
            parent: Some(parent_idx),
            verifier_pass: false,
            risk_score: 0.0,
            depth,
            expanded: false,
            pruned: false,
        }
    }

    /// Average value (exploitation term).
    fn avg_value(&self) -> f64 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_value / self.visits as f64
    }

    /// UCT score for selection. Higher = more promising + under-explored.
    fn uct(&self, parent_visits: u32, c: f32) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.avg_value();
        let exploration = c as f64 * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        exploitation + exploration
    }
}

// ─── Search Result ───────────────────────────────────────────────────────────

/// The result of a `think()` cycle.
#[derive(Debug, Clone, Serialize)]
pub struct ThinkResult {
    /// The best draft selected after deliberation.
    pub best_draft: String,
    /// Score of the best draft [0.0, 1.3].
    pub best_score: f32,
    /// Total nodes in the thought tree.
    pub nodes_explored: usize,
    /// How many branches were pruned (hallucination / low quality).
    pub branches_pruned: usize,
    /// Depth of the best node in the tree.
    pub best_depth: u32,
    /// Whether the best draft passed the WASI verifier.
    pub verifier_pass: bool,
    /// Total time spent thinking.
    pub latency_ms: u64,

    /// Number of thinking iterations completed.
    pub iterations_completed: usize,
    /// Chronological log of the thinking process (e.g. "Selected node 5 -> Expanded 3 children").
    pub process_log: Vec<String>,
}

// ─── TreeSearchAgent ─────────────────────────────────────────────────────────

/// The System 2 reasoning engine. Created per-request, stateless.
pub struct TreeSearchAgent {
    arena: Vec<ThoughtNode>,
    root_id: usize,
    config: ThinkConfig,
    log: Vec<String>,
}

impl TreeSearchAgent {
    /// Create a new agent with a root prompt.
    pub fn new(prompt: &str, config: ThinkConfig) -> Self {
        let mut arena = Vec::with_capacity(128);
        arena.push(ThoughtNode::root(prompt.to_string()));
        Self {
            arena,
            root_id: 0,
            config,
            log: Vec::new(),
        }
    }

    /// The main thinking loop — System 2 deliberation.
    ///
    /// Runs `config.iterations` cycles of SELECT→EXPAND→EVALUATE→BACKPROP.
    /// After all iterations, returns the highest-scoring draft.
    pub fn think(
        &mut self,
        runtime: &ModelRuntime,
        verifier: &Verifier,
        world_model: &WorldModel,
    ) -> Result<ThinkResult> {
        let started = Instant::now();
        let mut iterations_done = 0;
        let mut branches_pruned = 0;

        for _iter in 0..self.config.iterations {
            // ── 1. SELECT: Walk from root to most promising leaf via UCT ──
            let leaf_id = self.select(self.root_id);

            // Skip pruned or already-expanded-and-deep nodes.
            if self.arena[leaf_id].pruned {
                self.log(format!(
                    "Selected node {} but it was pruned (logic error?).",
                    leaf_id
                ));
                continue;
            }
            if self.arena[leaf_id].depth >= self.config.max_depth && self.arena[leaf_id].expanded {
                // Already at max depth and expanded — just re-evaluate.
                let score = self.evaluate(leaf_id, verifier, world_model);
                self.backpropagate(leaf_id, score);
                iterations_done += 1;
                continue;
            }

            // ── 2. EXPAND: Generate N candidate drafts from this node ────
            if !self.arena[leaf_id].expanded {
                self.log(format!(
                    "Expanding node {} (depth {})",
                    leaf_id, self.arena[leaf_id].depth
                ));
                self.expand(leaf_id, runtime)?;
            }

            // ── 3. EVALUATE: Score each new child ────────────────────────
            let child_ids: Vec<usize> = self.arena[leaf_id].children.clone();
            for &child_id in &child_ids {
                let score = self.evaluate(child_id, verifier, world_model);

                // ── 4. BACKPROPAGATE: Push score up to root ──────────────
                self.backpropagate(child_id, score);
            }

            // ── 5. PRUNE: Kill low-quality branches ──────────────────────
            let pruned_now = self.prune();
            branches_pruned += pruned_now;
            if pruned_now > 0 {
                self.log(format!("Pruned {} branches this iteration.", pruned_now));
            }

            iterations_done += 1;
        }

        // Pick the best node across the entire tree (not just root's children).
        let (best_id, best_score) = self.find_best_leaf();

        Ok(ThinkResult {
            best_draft: self.arena[best_id].state.clone(),
            best_score: best_score as f32,
            nodes_explored: self.arena.len(),
            branches_pruned,
            best_depth: self.arena[best_id].depth,
            verifier_pass: self.arena[best_id].verifier_pass,
            latency_ms: started.elapsed().as_millis() as u64,
            iterations_completed: iterations_done,
            process_log: self.log.clone(),
        })
    }

    /// Streaming version of `think()`. Emits `ThinkingEvent`s via `sender`.
    /// Streaming version of `think()`. Emits `ThinkingEvent`s via `sender`.
    pub fn think_stream(
        &mut self,
        runtime: &ModelRuntime,
        verifier: &Verifier,
        world_model: &WorldModel,
        sender: mpsc::Sender<Result<ThinkingEvent, anyhow::Error>>,
    ) -> Result<ThinkResult> {
        let started = Instant::now();
        let mut iterations_done = 0;
        let mut branches_pruned = 0;

        for _iter in 0..self.config.iterations {
            let leaf_id = self.select(self.root_id);

            // Emit thought event
            let _ = sender.blocking_send(Ok(ThinkingEvent::Thought(format!(
                "Selected node {} (depth {}).",
                leaf_id, self.arena[leaf_id].depth
            ))));

            if self.arena[leaf_id].pruned {
                continue;
            }
            if self.arena[leaf_id].depth >= self.config.max_depth && self.arena[leaf_id].expanded {
                let score = self.evaluate(leaf_id, verifier, world_model);
                self.backpropagate(leaf_id, score);
                iterations_done += 1;
                continue;
            }

            if !self.arena[leaf_id].expanded {
                let _ = sender.blocking_send(Ok(ThinkingEvent::Thought(format!(
                    "Expanding node {}...",
                    leaf_id
                ))));

                self.expand(leaf_id, runtime)?;
            }

            let child_ids: Vec<usize> = self.arena[leaf_id].children.clone();
            for &child_id in &child_ids {
                let score = self.evaluate(child_id, verifier, world_model);
                self.backpropagate(child_id, score);
            }

            let pruned_now = self.prune();
            branches_pruned += pruned_now;
            if pruned_now > 0 {
                let _ = sender.blocking_send(Ok(ThinkingEvent::Thought(format!(
                    "Pruned {} branches.",
                    pruned_now
                ))));
            }

            iterations_done += 1;
        }

        let (best_id, best_score) = self.find_best_leaf();
        let best_draft = self.arena[best_id].state.clone();

        // Emit final token events (simulated streaming of the best draft)
        // In a real scenario, we might re-generate or just chunk it.
        // For Draft-Level MCTS, we have the full text.
        let chunk_size = 4;
        for chunk in best_draft.chars().collect::<Vec<_>>().chunks(chunk_size) {
            let chunk_str: String = chunk.iter().collect();
            let _ = sender.blocking_send(Ok(ThinkingEvent::Token(chunk_str)));
            // simulated typing effect in sync context:
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let result = ThinkResult {
            best_draft: best_draft.clone(),
            best_score: best_score as f32,
            nodes_explored: self.arena.len(),
            branches_pruned,
            best_depth: self.arena[best_id].depth,
            verifier_pass: self.arena[best_id].verifier_pass,
            latency_ms: started.elapsed().as_millis() as u64,
            iterations_completed: iterations_done,
            process_log: self.log.clone(),
        };

        // Emit Summary event so client gets metadata
        let trace = crate::models::ThinkTrace {
            nodes_explored: result.nodes_explored,
            best_score: result.best_score,
            branches_pruned: result.branches_pruned,
            think_latency_ms: result.latency_ms,
            verifier_pass: result.verifier_pass,
            iterations_completed: result.iterations_completed,
            best_depth: result.best_depth,
            process_log: result.process_log.clone(),
        };
        let _ = sender.blocking_send(Ok(ThinkingEvent::Summary(trace)));

        let _ = sender.blocking_send(Ok(ThinkingEvent::Done));

        Ok(result)
    }

    fn log(&mut self, msg: impl Into<String>) {
        if self.log.len() < 100 {
            self.log.push(msg.into());
        }
    }

    // ── SELECT ───────────────────────────────────────────────────────────

    /// Walk from `node_id` to the most promising unpruned leaf using UCT.
    fn select(&self, node_id: usize) -> usize {
        let node = &self.arena[node_id];

        // If no children or not expanded, this is our leaf.
        if node.children.is_empty() || !node.expanded {
            return node_id;
        }

        // Filter out pruned children.
        let live_children: Vec<usize> = node
            .children
            .iter()
            .copied()
            .filter(|&c| !self.arena[c].pruned)
            .collect();

        if live_children.is_empty() {
            return node_id;
        }

        let parent_visits = node.visits.max(1);
        let best_child = live_children
            .iter()
            .max_by(|&&a, &&b| {
                let ua = self.arena[a].uct(parent_visits, self.config.exploration_c);
                let ub = self.arena[b].uct(parent_visits, self.config.exploration_c);
                ua.partial_cmp(&ub).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(node_id);

        self.select(best_child)
    }

    // ── EXPAND ───────────────────────────────────────────────────────────

    /// Generate `candidates_per_expand` drafts from the node's state.
    fn expand(&mut self, node_id: usize, runtime: &ModelRuntime) -> Result<()> {
        if self.arena[node_id].expanded {
            return Ok(());
        }

        let prompt = &self.arena[node_id].state;
        let depth = self.arena[node_id].depth;

        // Build the expansion prompt.
        // For root: use prompt directly.
        // For deeper nodes: ask the model to refine/improve the draft.
        let expand_prompt = if depth == 0 {
            prompt.clone()
        } else {
            format!(
                "Improve and refine the following draft. Fix any errors, \
                 improve clarity, and ensure correctness:\n\n{prompt}"
            )
        };

        let mut child_indices = Vec::new();

        for candidate_idx in 0..self.config.candidates_per_expand {
            // Generate a candidate draft.
            // Vary token count slightly per candidate to get diversity.
            let max_tokens = self
                .config
                .max_tokens_per_draft
                .saturating_sub(candidate_idx * 8)
                .max(32);

            match runtime.infer(&expand_prompt, max_tokens) {
                Ok(response) => {
                    let draft = response.text.trim().to_string();
                    if draft.is_empty() {
                        continue;
                    }

                    let child = ThoughtNode::child(draft, node_id, depth + 1);
                    let child_idx = self.arena.len();
                    self.arena.push(child);
                    child_indices.push(child_idx);
                }
                Err(e) => {
                    let msg = format!("Model inference failed: {}", e);
                    self.log(msg.clone());
                    eprintln!("{}", msg);
                    continue;
                }
            }
        }

        if child_indices.is_empty() {
            let msg = format!(
                "Expansion failed for node {}: no valid drafts generated.",
                node_id
            );
            self.log(msg.clone());
            eprintln!("{}", msg);
        } else {
            self.log(format!(
                "Generated {} new drafts from node {}.",
                child_indices.len(),
                node_id
            ));
            self.arena[node_id].expanded = true;
        }

        self.arena[node_id].children = child_indices;
        Ok(())
    }

    // ── EVALUATE ─────────────────────────────────────────────────────────

    /// Score a node using the WASI verifier + world model.
    /// Returns a score in [0.0, 1.3].
    fn evaluate(&mut self, node_id: usize, verifier: &Verifier, world_model: &WorldModel) -> f64 {
        let draft = &self.arena[node_id].state;
        if draft.trim().is_empty() {
            return 0.0;
        }

        // World model: risk + confidence.
        let sim = world_model.simulate(draft);
        let confidence = sim.confidence as f64;
        let risk = sim.risk_score as f64;

        // WASI verifier: pass/fail + violations.
        let verifier_result = verifier.check(&VerifierRequest {
            patch_ir: draft.to_string(),
            max_loop_iters: Some(2048),
            side_effect_budget: Some(3),
            timeout_ms: Some(50),
        });

        // Store results on the node.
        self.arena[node_id].verifier_pass = verifier_result.pass;
        self.arena[node_id].risk_score = sim.risk_score;

        // Composite score.
        let base = confidence * (1.0 - risk);
        let verifier_bonus = if verifier_result.pass { 0.3 } else { -0.1 };
        let violation_penalty = verifier_result.violations.len() as f64 * 0.05;

        // Depth bonus: slightly prefer deeper refinements (they've been improved).
        let depth_bonus = (self.arena[node_id].depth as f64) * 0.02;

        (base + verifier_bonus - violation_penalty + depth_bonus).clamp(0.0, 1.3)
    }

    // ── BACKPROPAGATE ────────────────────────────────────────────────────

    /// Update visit count and total value from node back to root.
    fn backpropagate(&mut self, node_id: usize, score: f64) {
        let mut current = Some(node_id);
        while let Some(id) = current {
            self.arena[id].visits += 1;
            self.arena[id].total_value += score;
            current = self.arena[id].parent;
        }
    }

    // ── PRUNE ────────────────────────────────────────────────────────────

    /// Prune branches with average score below the threshold.
    /// Returns the number of branches pruned this round.
    fn prune(&mut self) -> usize {
        let mut pruned_count = 0;
        let threshold = self.config.prune_threshold as f64;

        for i in 1..self.arena.len() {
            // Skip already pruned, root, or unvisited nodes.
            if self.arena[i].pruned || self.arena[i].visits == 0 {
                continue;
            }

            let avg = self.arena[i].avg_value();
            if avg < threshold && self.arena[i].visits >= 2 {
                self.arena[i].pruned = true;
                // Recursively prune all descendants.
                self.prune_subtree(i);
                pruned_count += 1;
            }
        }

        pruned_count
    }

    /// Mark all descendants of a node as pruned.
    fn prune_subtree(&mut self, node_id: usize) {
        let children: Vec<usize> = self.arena[node_id].children.clone();
        for child_id in children {
            self.arena[child_id].pruned = true;
            self.prune_subtree(child_id);
        }
    }

    // ── BEST LEAF ────────────────────────────────────────────────────────

    /// Find the best non-root, non-pruned node with the highest average value.
    fn find_best_leaf(&self) -> (usize, f64) {
        let mut best_id = self.root_id;
        let mut best_score = f64::NEG_INFINITY;

        for (i, node) in self.arena.iter().enumerate() {
            // Skip root, pruned, and unvisited.
            if i == self.root_id || node.pruned || node.visits == 0 {
                continue;
            }

            let avg = node.avg_value();

            // Prefer verifier-passing nodes.
            let adjusted = if node.verifier_pass { avg + 0.05 } else { avg };

            if adjusted > best_score {
                best_score = adjusted;
                best_id = i;
            }
        }

        // If no good node found (e.g., no expansion succeeded), fall back to root.
        if best_id == self.root_id {
            return (self.root_id, 0.0);
        }

        (best_id, best_score)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thought_node_root_has_depth_zero() {
        let root = ThoughtNode::root("test prompt".into());
        assert_eq!(root.depth, 0);
        assert!(root.parent.is_none());
        assert!(!root.expanded);
        assert!(!root.pruned);
    }

    #[test]
    fn thought_node_child_links_parent() {
        let child = ThoughtNode::child("draft A".into(), 0, 1);
        assert_eq!(child.parent, Some(0));
        assert_eq!(child.depth, 1);
    }

    #[test]
    fn uct_unvisited_is_infinity() {
        let node = ThoughtNode::root("test".into());
        assert!(node.uct(10, 1.414).is_infinite());
    }

    #[test]
    fn uct_exploitation_dominates_at_high_visits() {
        let mut good = ThoughtNode::child("good".into(), 0, 1);
        good.visits = 100;
        good.total_value = 90.0; // avg 0.9

        let mut bad = ThoughtNode::child("bad".into(), 0, 1);
        bad.visits = 100;
        bad.total_value = 20.0; // avg 0.2

        // High visits → exploitation dominates → good > bad.
        let uct_good = good.uct(200, 1.414);
        let uct_bad = bad.uct(200, 1.414);
        assert!(uct_good > uct_bad, "good={uct_good} should > bad={uct_bad}");
    }

    #[test]
    fn backpropagation_reaches_root() {
        let mut agent = TreeSearchAgent::new("prompt", ThinkConfig::default());

        // Manually add a child.
        let child = ThoughtNode::child("draft".into(), 0, 1);
        agent.arena.push(child);
        agent.arena[0].children.push(1);
        agent.arena[0].expanded = true;

        // Backprop score from child.
        agent.backpropagate(1, 0.8);

        assert_eq!(agent.arena[1].visits, 1);
        assert!((agent.arena[1].total_value - 0.8).abs() < 0.001);
        assert_eq!(agent.arena[0].visits, 1); // Root also visited.
        assert!((agent.arena[0].total_value - 0.8).abs() < 0.001);
    }

    #[test]
    fn pruning_removes_low_score_branches() {
        let config = ThinkConfig {
            prune_threshold: 0.3,
            ..ThinkConfig::default()
        };
        let mut agent = TreeSearchAgent::new("prompt", config);

        // Add child with low score.
        let mut bad_child = ThoughtNode::child("bad draft".into(), 0, 1);
        bad_child.visits = 3;
        bad_child.total_value = 0.3; // avg = 0.1 < threshold 0.3
        agent.arena.push(bad_child);
        agent.arena[0].children.push(1);

        // Add grandchild.
        let grandchild = ThoughtNode::child("grandchild".into(), 1, 2);
        agent.arena.push(grandchild);
        agent.arena[1].children.push(2);

        let pruned = agent.prune();
        assert_eq!(pruned, 1);
        assert!(agent.arena[1].pruned);
        assert!(agent.arena[2].pruned, "grandchild should also be pruned");
    }

    #[test]
    fn select_prefers_unvisited() {
        let mut agent = TreeSearchAgent::new("prompt", ThinkConfig::default());

        // Add two children: one visited, one not.
        let mut visited = ThoughtNode::child("visited".into(), 0, 1);
        visited.visits = 5;
        visited.total_value = 2.0;
        agent.arena.push(visited);

        let unvisited = ThoughtNode::child("unvisited".into(), 0, 1);
        agent.arena.push(unvisited);

        agent.arena[0].children = vec![1, 2];
        agent.arena[0].expanded = true;

        let selected = agent.select(0);
        assert_eq!(
            selected, 2,
            "should select unvisited child (UCT = infinity)"
        );
    }

    #[test]
    fn select_skips_pruned() {
        let mut agent = TreeSearchAgent::new("prompt", ThinkConfig::default());

        let mut pruned_child = ThoughtNode::child("pruned".into(), 0, 1);
        pruned_child.pruned = true;
        agent.arena.push(pruned_child);

        let alive = ThoughtNode::child("alive".into(), 0, 1);
        agent.arena.push(alive);

        agent.arena[0].children = vec![1, 2];
        agent.arena[0].expanded = true;

        let selected = agent.select(0);
        assert_eq!(selected, 2, "should skip pruned child");
    }

    #[test]
    fn config_defaults_are_sensible() {
        let config = ThinkConfig::default();
        assert_eq!(config.iterations, 50);
        assert_eq!(config.candidates_per_expand, 3);
        assert!(config.exploration_c > 1.0);
        assert!(config.prune_threshold > 0.0 && config.prune_threshold < 0.5);
        assert!(config.max_depth >= 2);
    }

    #[test]
    fn find_best_leaf_prefers_verifier_pass() {
        let mut agent = TreeSearchAgent::new("prompt", ThinkConfig::default());

        // Child A: high score, no verifier pass.
        let mut a = ThoughtNode::child("draft A".into(), 0, 1);
        a.visits = 5;
        a.total_value = 4.0; // avg 0.80
        a.verifier_pass = false;
        agent.arena.push(a);

        // Child B: slightly lower score, but verifier passes.
        let mut b = ThoughtNode::child("draft B".into(), 0, 1);
        b.visits = 5;
        b.total_value = 3.9; // avg 0.78, but +0.05 bonus → 0.83
        b.verifier_pass = true;
        agent.arena.push(b);

        agent.arena[0].children = vec![1, 2];

        let (best_id, _) = agent.find_best_leaf();
        assert_eq!(best_id, 2, "should prefer verifier-passing draft B");
    }
}
