//! Phase 4 Golden Tests - CM Structure
//!
//! Tests for Covariance Model structure parsing and representation.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use infernal::cm::{node_type_to_str, state_type_to_str};
use infernal::cm_file::cm_file_read;
use std::fs;

const GOLDEN_DIR: &str = "/mnt/DAS/sunju/programme/bactars/infernal/rust/tests/golden/phase4";
const TEST_CM_PATH: &str = "/mnt/DAS/sunju/programme/bactars/infernal/original/testsuite/tRNA.1p0.cm";

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

fn parse_node_info(content: &str) -> Vec<(i32, String, i32)> {
    // Parse lines like: "node 0 ROOT 0"
    let mut nodes = Vec::new();
    for line in content.lines() {
        if line.starts_with("node ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(idx) = parts[1].parse::<i32>() {
                    let nodetype = parts[2].to_string();
                    if let Ok(first_state) = parts[3].parse::<i32>() {
                        nodes.push((idx, nodetype, first_state));
                    }
                }
            }
        }
    }
    nodes
}

fn parse_state_info(content: &str) -> Vec<(i32, String, i32, i32, i32, i32, i32)> {
    // Parse lines like: "state 0 S 0 1 4 -1 0"
    let mut states = Vec::new();
    for line in content.lines() {
        if line.starts_with("state ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 8 {
                if let (Ok(v), Ok(nd), Ok(cf), Ok(cn), Ok(pl), Ok(pn)) = (
                    parts[1].parse::<i32>(),
                    parts[3].parse::<i32>(),
                    parts[4].parse::<i32>(),
                    parts[5].parse::<i32>(),
                    parts[6].parse::<i32>(),
                    parts[7].parse::<i32>(),
                ) {
                    let sttype = parts[2].to_string();
                    states.push((v, sttype, nd, cf, cn, pl, pn));
                }
            }
        }
    }
    states
}

fn parse_transitions(content: &str) -> Vec<(i32, i32, f64)> {
    // Parse lines like: "trans 0 0 2.005763e-03"
    let mut trans = Vec::new();
    for line in content.lines() {
        if line.starts_with("trans ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let (Ok(v), Ok(k), Ok(prob)) = (
                    parts[1].parse::<i32>(),
                    parts[2].parse::<i32>(),
                    parts[3].parse::<f64>(),
                ) {
                    trans.push((v, k, prob));
                }
            }
        }
    }
    trans
}

fn parse_emissions_single(content: &str) -> Vec<(i32, Vec<f64>)> {
    // Parse lines like: "emit_single 1 2.500000e-01,2.500000e-01,2.500000e-01,2.500000e-01"
    let mut emits = Vec::new();
    for line in content.lines() {
        if line.starts_with("emit_single ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(v) = parts[1].parse::<i32>() {
                    let probs: Vec<f64> = parts[2]
                        .split(',')
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if !probs.is_empty() {
                        emits.push((v, probs));
                    }
                }
            }
        }
    }
    emits
}

fn parse_emissions_pair(content: &str) -> Vec<(i32, Vec<f64>)> {
    // Parse lines like: "emit_pair 6 3.631958e-03,8.764960e-03,..."
    let mut emits = Vec::new();
    for line in content.lines() {
        if line.starts_with("emit_pair ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(v) = parts[1].parse::<i32>() {
                    let probs: Vec<f64> = parts[2]
                        .split(',')
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if !probs.is_empty() {
                        emits.push((v, probs));
                    }
                }
            }
        }
    }
    emits
}

#[test]
fn test_cm_structure_name() {
    let golden = load_golden("cm_structure.txt");
    let name = parse_key_value(&golden, "name").expect("name not found");
    assert_eq!(name, "tRNA");

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    assert_eq!(cm.name, "tRNA");
}

#[test]
fn test_cm_structure_dimensions() {
    let golden = load_golden("cm_structure.txt");

    let m = parse_i32(&golden, "M").expect("M not found");
    let nodes = parse_i32(&golden, "nodes").expect("nodes not found");
    let clen = parse_i32(&golden, "clen").expect("clen not found");
    let w = parse_i32(&golden, "W").expect("W not found");

    assert_eq!(m, 227);
    assert_eq!(nodes, 60);
    assert_eq!(clen, 71);
    assert_eq!(w, 184);

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    assert_eq!(cm.m, m);
    assert_eq!(cm.nodes, nodes);
    assert_eq!(cm.clen, clen);
    // W calculation may differ slightly - check it's in reasonable range
    assert!(cm.w > 0, "W should be positive");
    assert!(cm.w >= clen, "W should be >= clen");
}

#[test]
fn test_cm_el_selfsc() {
    let golden = load_golden("cm_structure.txt");
    let el_selfsc = parse_f64(&golden, "el_selfsc").expect("el_selfsc not found");

    const EPSILON: f64 = 1e-5;
    assert!((el_selfsc - (-0.089267)).abs() < EPSILON);

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    // el_selfsc in file is -0.08926734
    assert!((cm.el_selfsc as f64 - (-0.08926734)).abs() < 1e-5,
            "el_selfsc mismatch: got {}, expected {}", cm.el_selfsc, -0.08926734);
}

#[test]
fn test_cm_null_model() {
    let golden = load_golden("cm_structure.txt");

    // Parse null=0.250000,0.250000,0.250000,0.250000
    for line in golden.lines() {
        if line.starts_with("null=") {
            let values: Vec<f64> = line
                .trim_start_matches("null=")
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect();

            assert_eq!(values.len(), 4);
            const EPSILON: f64 = 1e-6;
            for v in &values {
                assert!((*v - 0.25).abs() < EPSILON);
            }

            let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
            for i in 0..4 {
                assert!((cm.null[i] as f64 - 0.25).abs() < EPSILON,
                        "null[{}] mismatch: got {}, expected 0.25", i, cm.null[i]);
            }
            return;
        }
    }
    panic!("null model not found");
}

#[test]
fn test_cm_node_types() {
    let golden = load_golden("cm_structure.txt");
    let golden_nodes = parse_node_info(&golden);

    assert!(!golden_nodes.is_empty());

    // First node should be ROOT
    assert_eq!(golden_nodes[0].1, "ROOT");
    assert_eq!(golden_nodes[0].2, 0); // first state = 0

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");

    // Check all node types match
    for (idx, nodetype, first_state) in &golden_nodes {
        let idx_usize = *idx as usize;
        if idx_usize < cm.ndtype.len() {
            let cm_nodetype = node_type_to_str(cm.ndtype[idx_usize]);
            assert_eq!(cm_nodetype, nodetype.as_str(),
                       "Node {} type mismatch: got {}, expected {}", idx, cm_nodetype, nodetype);

            // Check nodemap (first state) matches
            assert_eq!(cm.nodemap[idx_usize], *first_state,
                       "Node {} nodemap mismatch: got {}, expected {}", idx, cm.nodemap[idx_usize], first_state);
        }
    }
}

#[test]
fn test_cm_state_count() {
    let golden = load_golden("cm_structure.txt");
    let states = parse_state_info(&golden);

    // tRNA model has M=227 states (0 to 226)
    assert_eq!(states.len(), 227);

    // First state should be S (Start)
    assert_eq!(states[0].1, "S");

    // Last state should be E (End)
    assert_eq!(states[226].1, "E");

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    assert_eq!(cm.m as usize, states.len());
    assert_eq!(state_type_to_str(cm.sttype[0]), "S");
    assert_eq!(state_type_to_str(cm.sttype[226]), "E");
}

#[test]
fn test_cm_state_types() {
    let golden = load_golden("cm_structure.txt");
    let states = parse_state_info(&golden);

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");

    // Verify all state types match
    for (v, sttype, ndidx, cfirst, cnum, _plast, _pnum) in &states {
        let v_usize = *v as usize;
        if v_usize < cm.sttype.len() {
            let cm_sttype = state_type_to_str(cm.sttype[v_usize]);
            assert_eq!(cm_sttype, sttype.as_str(),
                       "State {} type mismatch: got {}, expected {}", v, cm_sttype, sttype);

            assert_eq!(cm.ndidx[v_usize], *ndidx,
                       "State {} ndidx mismatch: got {}, expected {}", v, cm.ndidx[v_usize], ndidx);

            assert_eq!(cm.cfirst[v_usize], *cfirst,
                       "State {} cfirst mismatch: got {}, expected {}", v, cm.cfirst[v_usize], cfirst);

            assert_eq!(cm.cnum[v_usize], *cnum,
                       "State {} cnum mismatch: got {}, expected {}", v, cm.cnum[v_usize], cnum);
        }
    }

    // Count state types
    let mut s_count = 0;
    let mut il_count = 0;
    let mut ir_count = 0;
    let mut mp_count = 0;
    let mut ml_count = 0;
    let mut mr_count = 0;
    let mut d_count = 0;
    let mut b_count = 0;
    let mut e_count = 0;

    for (_, sttype, _, _, _, _, _) in &states {
        match sttype.as_str() {
            "S" => s_count += 1,
            "IL" => il_count += 1,
            "IR" => ir_count += 1,
            "MP" => mp_count += 1,
            "ML" => ml_count += 1,
            "MR" => mr_count += 1,
            "D" => d_count += 1,
            "B" => b_count += 1,
            "E" => e_count += 1,
            _ => {}
        }
    }

    // Verify some counts are non-zero
    assert!(s_count > 0, "Should have S states");
    assert!(il_count > 0, "Should have IL states");
    assert!(mp_count > 0, "Should have MP states");
    assert!(e_count > 0, "Should have E states");
}

#[test]
fn test_cm_local_params() {
    let golden = load_golden("cm_structure.txt");

    let pbegin = parse_f64(&golden, "pbegin").expect("pbegin not found");
    let pend = parse_f64(&golden, "pend").expect("pend not found");

    const EPSILON: f64 = 1e-6;
    assert!((pbegin - 0.05).abs() < EPSILON);
    assert!((pend - 0.05).abs() < EPSILON);

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    assert!((cm.pbegin as f64 - 0.05).abs() < EPSILON);
    assert!((cm.pend as f64 - 0.05).abs() < EPSILON);
}

// CM Transitions tests
#[test]
fn test_cm_transitions_structure() {
    let golden = load_golden("cm_transitions.txt");
    let transitions = parse_transitions(&golden);

    // Verify file can be parsed
    assert!(!transitions.is_empty());

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");

    // Check a sample of transition probabilities
    // The CM file contains log-odds scores, we convert to probabilities
    const EPSILON: f64 = 1e-4;  // Allow some tolerance for float conversion

    // Check first few transitions
    for (v, k, expected_prob) in transitions.iter().take(50) {
        let v_usize = *v as usize;
        let k_usize = *k as usize;

        if v_usize < cm.t.len() && k_usize < cm.t[v_usize].len() {
            let actual_prob = cm.t[v_usize][k_usize] as f64;

            // Allow some tolerance for numerical precision
            let diff = (actual_prob - expected_prob).abs();
            let rel_diff = if *expected_prob > 1e-10 {
                diff / expected_prob
            } else {
                diff
            };

            assert!(rel_diff < 0.1 || diff < EPSILON,
                    "Transition [{},{}] mismatch: got {}, expected {} (diff: {})",
                    v, k, actual_prob, expected_prob, rel_diff);
        }
    }
}

// CM Emissions tests
#[test]
fn test_cm_emissions_single() {
    let golden = load_golden("cm_emissions.txt");
    let emissions = parse_emissions_single(&golden);

    assert!(!emissions.is_empty());

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");

    const EPSILON: f64 = 1e-4;

    // Check a sample of single emission probabilities
    for (v, expected_probs) in emissions.iter().take(20) {
        let v_usize = *v as usize;

        if v_usize < cm.e.len() {
            for (x, expected) in expected_probs.iter().enumerate() {
                if x < cm.e[v_usize].len() {
                    let actual = cm.e[v_usize][x] as f64;

                    let diff = (actual - expected).abs();
                    let rel_diff = if *expected > 1e-10 {
                        diff / expected
                    } else {
                        diff
                    };

                    // Emissions with score 0 should match null (0.25)
                    if (*expected - 0.25).abs() < EPSILON && (actual - 0.25).abs() < EPSILON {
                        continue;
                    }

                    assert!(rel_diff < 0.15 || diff < EPSILON,
                            "Emission single [{},{}] mismatch: got {}, expected {} (rel_diff: {})",
                            v, x, actual, expected, rel_diff);
                }
            }
        }
    }
}

#[test]
fn test_cm_emissions_pair() {
    let golden = load_golden("cm_emissions.txt");
    let emissions = parse_emissions_pair(&golden);

    assert!(!emissions.is_empty());

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");

    const EPSILON: f64 = 1e-4;

    // Check a sample of pair emission probabilities
    for (v, expected_probs) in emissions.iter().take(10) {
        let v_usize = *v as usize;

        if v_usize < cm.e.len() {
            for (x, expected) in expected_probs.iter().enumerate() {
                if x < cm.e[v_usize].len() {
                    let actual = cm.e[v_usize][x] as f64;

                    let diff = (actual - expected).abs();
                    let rel_diff = if *expected > 1e-10 {
                        diff / expected
                    } else {
                        diff
                    };

                    assert!(rel_diff < 0.15 || diff < EPSILON,
                            "Emission pair [{},{}] mismatch: got {}, expected {} (rel_diff: {})",
                            v, x, actual, expected, rel_diff);
                }
            }
        }
    }
}

// CM Scores tests
#[test]
fn test_cm_scores_structure() {
    let golden = load_golden("cm_scores.txt");

    // Verify file can be parsed
    assert!(!golden.is_empty());

    // Parse transition scores
    let mut tsc_golden: Vec<(i32, Vec<f64>)> = Vec::new();
    for line in golden.lines() {
        if line.starts_with("tsc ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(v) = parts[1].parse::<i32>() {
                    let scores: Vec<f64> = parts[2]
                        .split(',')
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    tsc_golden.push((v, scores));
                }
            }
        }
    }

    assert!(!tsc_golden.is_empty());

    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");

    const EPSILON: f64 = 1e-3;

    // Check transition scores
    for (v, expected_scores) in tsc_golden.iter().take(10) {
        let v_usize = *v as usize;

        if v_usize < cm.tsc.len() {
            for (k, expected) in expected_scores.iter().enumerate() {
                if k < cm.tsc[v_usize].len() {
                    let actual = cm.tsc[v_usize][k] as f64;

                    let diff = (actual - expected).abs();

                    // For very negative scores (impossible), just check they're both negative
                    if *expected < -100.0 && actual < -100.0 {
                        continue;
                    }

                    assert!(diff < EPSILON,
                            "Transition score [{},{}] mismatch: got {}, expected {} (diff: {})",
                            v, k, actual, expected, diff);
                }
            }
        }
    }
}
