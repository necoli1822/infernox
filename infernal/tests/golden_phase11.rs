//! Phase 11 Golden Tests - Alignment and Traceback
//!
//! Tests for alignment generation and traceback (parsetree) construction.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use std::fs;

const GOLDEN_DIR: &str = "tests/golden/phase11";

fn load_golden(filename: &str) -> String {
    fs::read_to_string(format!("{}/{}", GOLDEN_DIR, filename))
        .expect("Failed to load golden file")
}

fn parse_key_value(content: &str, key: &str) -> Option<String> {
    for line in content.lines() {
        if line.starts_with(&format!("{}=", key)) {
            return Some(line.split('=').nth(1)?.to_string());
        }
    }
    None
}

fn parse_f64(content: &str, key: &str) -> Option<f64> {
    parse_key_value(content, key)?.parse().ok()
}

fn parse_i32(content: &str, key: &str) -> Option<i32> {
    parse_key_value(content, key)?.parse().ok()
}

const EPSILON: f64 = 1e-4;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// Parse parsetree node entries
/// Format: node idx emitl=X emitr=Y state=Z mode=M nxtl=A nxtr=B prv=C
fn parse_parsetree_nodes(content: &str) -> Vec<ParseTreeNode> {
    let mut nodes = Vec::new();
    for line in content.lines() {
        if line.starts_with("node ") {
            let node = ParseTreeNode::from_line(line);
            if let Some(n) = node {
                nodes.push(n);
            }
        }
    }
    nodes
}

#[derive(Debug, Clone)]
struct ParseTreeNode {
    idx: i32,
    emitl: i32,
    emitr: i32,
    state: i32,
    mode: String,
    nxtl: i32,
    nxtr: i32,
    prv: i32,
}

impl ParseTreeNode {
    fn from_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 9 {
            return None;
        }

        // Parse: node X emitl=A emitr=B state=C mode=D nxtl=E nxtr=F prv=G
        let idx = parts[1].parse().ok()?;

        fn extract_val(s: &str) -> Option<i32> {
            s.split('=').last()?.parse().ok()
        }

        fn extract_str(s: &str) -> String {
            s.split('=').last().unwrap_or("").to_string()
        }

        Some(ParseTreeNode {
            idx,
            emitl: extract_val(parts[2])?,
            emitr: extract_val(parts[3])?,
            state: extract_val(parts[4])?,
            mode: extract_str(parts[5]),
            nxtl: extract_val(parts[6])?,
            nxtr: extract_val(parts[7])?,
            prv: extract_val(parts[8])?,
        })
    }
}

// =====================================================================
// Traceback Path Tests
// =====================================================================

#[test]
fn test_traceback_alignment_score() {
    let golden = load_golden("traceback_path.txt");

    let score = parse_f64(&golden, "alignment_score").expect("alignment_score not found");

    assert!(approx_eq(score, 48.0963), "Alignment score should be ~48.0963, got {}", score);
}

#[test]
fn test_traceback_parsetree_size() {
    let golden = load_golden("traceback_path.txt");

    let size = parse_i32(&golden, "parsetree_size").expect("parsetree_size not found");

    assert_eq!(size, 65, "Parsetree should have 65 nodes");
}

#[test]
fn test_traceback_is_standard() {
    let golden = load_golden("traceback_path.txt");

    let is_std = parse_i32(&golden, "is_standard").expect("is_standard not found");

    assert_eq!(is_std, 1, "Should be standard (not truncated) alignment");
}

#[test]
fn test_traceback_no_penalty() {
    let golden = load_golden("traceback_path.txt");

    let penalty = parse_f64(&golden, "trpenalty").expect("trpenalty not found");

    assert!(approx_eq(penalty, 0.0), "No truncation penalty for standard alignment");
}

// =====================================================================
// Parsetree Structure Tests
// =====================================================================

#[test]
fn test_traceback_parsetree_root() {
    let golden = load_golden("traceback_path.txt");
    let nodes = parse_parsetree_nodes(&golden);

    assert!(!nodes.is_empty(), "Should have parsetree nodes");

    // First node should be the root
    let root = &nodes[0];
    assert_eq!(root.idx, 0);
    assert_eq!(root.state, 0, "Root should be state 0 (S)");
    assert_eq!(root.emitl, 1, "Root should start at position 1");
    assert_eq!(root.emitr, 76, "Root should end at position 76");
    assert_eq!(root.prv, -1, "Root has no parent");
}

#[test]
fn test_traceback_parsetree_mode() {
    let golden = load_golden("traceback_path.txt");
    let nodes = parse_parsetree_nodes(&golden);

    // All nodes should have mode J (Joint) for standard alignment
    for node in &nodes {
        assert_eq!(
            node.mode, "J",
            "Node {} should have mode J, got {}",
            node.idx, node.mode
        );
    }
}

#[test]
fn test_traceback_parsetree_connectivity() {
    let golden = load_golden("traceback_path.txt");
    let nodes = parse_parsetree_nodes(&golden);

    // Build index for lookup
    let idx_map: std::collections::HashMap<i32, &ParseTreeNode> =
        nodes.iter().map(|n| (n.idx, n)).collect();

    // Verify parent-child relationships
    for node in &nodes {
        if node.prv >= 0 {
            assert!(
                idx_map.contains_key(&node.prv),
                "Node {} parent {} not found",
                node.idx,
                node.prv
            );
        }

        if node.nxtl >= 0 {
            assert!(
                idx_map.contains_key(&node.nxtl),
                "Node {} left child {} not found",
                node.idx,
                node.nxtl
            );
        }

        if node.nxtr >= 0 {
            assert!(
                idx_map.contains_key(&node.nxtr),
                "Node {} right child {} not found",
                node.idx,
                node.nxtr
            );
        }
    }
}

#[test]
fn test_traceback_parsetree_emits() {
    let golden = load_golden("traceback_path.txt");
    let nodes = parse_parsetree_nodes(&golden);

    // Verify emit positions are within sequence bounds (1-76)
    for node in &nodes {
        assert!(
            node.emitl >= 1 && node.emitl <= 76,
            "Node {} emitl {} out of bounds",
            node.idx,
            node.emitl
        );
        assert!(
            node.emitr >= 1 && node.emitr <= 76,
            "Node {} emitr {} out of bounds",
            node.idx,
            node.emitr
        );
        // Note: emitl can be > emitr for end states (no emission)
    }
}

// =====================================================================
// State Type Tests
// =====================================================================

#[test]
fn test_traceback_state_types() {
    let golden = load_golden("traceback_path.txt");

    // Parse state type annotations
    // Format: state X type=T node=N
    let mut state_types: Vec<(i32, String, i32)> = Vec::new();
    for line in golden.lines() {
        if line.starts_with("state ") && line.contains("type=") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(state) = parts[1].parse::<i32>() {
                    let state_type = parts[2]
                        .split('=')
                        .last()
                        .unwrap_or("")
                        .to_string();
                    let node = parts[3]
                        .split('=')
                        .last()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(-1);
                    state_types.push((state, state_type, node));
                }
            }
        }
    }

    // Should have various state types in the trace
    let types: Vec<&str> = state_types.iter().map(|(_, t, _)| t.as_str()).collect();

    assert!(types.contains(&"S"), "Should have S (Start) state");
    assert!(types.contains(&"MP"), "Should have MP (Match Pair) states");
    assert!(types.contains(&"ML"), "Should have ML (Match Left) states");
    assert!(types.contains(&"B"), "Should have B (Bifurcation) states");
}

// =====================================================================
// Alignment Output Tests
// =====================================================================

#[test]
fn test_alignment_output_structure() {
    let golden = load_golden("alignment_output.txt");

    // Verify file has content
    assert!(!golden.is_empty());

    // TODO: Parse and verify alignment format
}

// =====================================================================
// Bifurcation Tests
// =====================================================================

#[test]
fn test_traceback_bifurcation_nodes() {
    let golden = load_golden("traceback_path.txt");
    let nodes = parse_parsetree_nodes(&golden);

    // Find bifurcation nodes (nxtr != -1)
    let bif_nodes: Vec<_> = nodes.iter().filter(|n| n.nxtr != -1).collect();

    // tRNA model has bifurcations
    assert!(!bif_nodes.is_empty(), "Should have bifurcation nodes");

    // Each bifurcation should have two children
    for bif in &bif_nodes {
        assert!(bif.nxtl >= 0, "Bifurcation should have left child");
        assert!(bif.nxtr >= 0, "Bifurcation should have right child");
    }
}

#[test]
fn test_traceback_leaf_nodes() {
    let golden = load_golden("traceback_path.txt");
    let nodes = parse_parsetree_nodes(&golden);

    // Leaf nodes have nxtl == -1
    let leaf_nodes: Vec<_> = nodes.iter().filter(|n| n.nxtl == -1).collect();

    // Should have multiple leaf nodes (one for each END state reached)
    assert!(leaf_nodes.len() >= 3, "Should have multiple leaf nodes (END states)");

    // Leaf nodes should also have nxtr == -1
    for leaf in &leaf_nodes {
        assert_eq!(
            leaf.nxtr, -1,
            "Leaf node {} should have nxtr == -1",
            leaf.idx
        );
    }
}
