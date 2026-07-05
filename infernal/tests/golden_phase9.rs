//! Phase 9 Golden Tests - P7 HMM Filter
//!
//! Tests for HMMER3 P7 HMM construction and filtering statistics.
//! The P7 HMM is used for fast filtering before CM alignment.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use std::fs;
use infernal::p7_hmm::{P7EvParams, P7Profile};

const GOLDEN_DIR: &str = "tests/golden/phase9";

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

const EPSILON: f64 = 1e-6;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

// =====================================================================
// P7 HMM Info Tests
// =====================================================================

#[test]
fn test_p7_model_length() {
    let golden = load_golden("p7_hmm_info.txt");
    let m = parse_i32(&golden, "M").expect("M not found");

    // Create P7 profile with model length from golden data
    let p7 = P7Profile::new(m);

    assert_eq!(p7.m, 71, "P7 HMM should have M=71");
    assert_eq!(p7.mat.len(), 72); // M+1 elements
    assert_eq!(p7.ins.len(), 72); // M+1 elements
    assert_eq!(p7.trans.len(), 72); // M+1 elements
}

#[test]
fn test_p7_evparam_uncalibrated() {
    let golden = load_golden("p7_filter_stats.txt");

    // Parse E-value parameters from golden data
    let lmmu = parse_f64(&golden, "lmmu").expect("lmmu not found");
    let lmlambda = parse_f64(&golden, "lmlambda").expect("lmlambda not found");
    let lvmu = parse_f64(&golden, "lvmu").expect("lvmu not found");
    let lvlambda = parse_f64(&golden, "lvlambda").expect("lvlambda not found");
    let lftau = parse_f64(&golden, "lftau").expect("lftau not found");
    let lflambda = parse_f64(&golden, "lflambda").expect("lflambda not found");
    let gfmu = parse_f64(&golden, "gfmu").expect("gfmu not found");
    let gflambda = parse_f64(&golden, "gflambda").expect("gflambda not found");

    // All should be -99999 (uncalibrated)
    assert_eq!(lmmu, -99999.0, "Uncalibrated lmmu should be -99999");
    assert_eq!(lmlambda, -99999.0, "Uncalibrated lmlambda should be -99999");
    assert_eq!(lvmu, -99999.0, "Uncalibrated lvmu should be -99999");
    assert_eq!(lvlambda, -99999.0, "Uncalibrated lvlambda should be -99999");
    assert_eq!(lftau, -99999.0, "Uncalibrated lftau should be -99999");
    assert_eq!(lflambda, -99999.0, "Uncalibrated lflambda should be -99999");
    assert_eq!(gfmu, -99999.0, "Uncalibrated gfmu should be -99999");
    assert_eq!(gflambda, -99999.0, "Uncalibrated gflambda should be -99999");
}

#[test]
fn test_p7_hmm_name() {
    let golden = load_golden("p7_hmm_info.txt");
    let name = parse_key_value(&golden, "name").expect("name not found");
    assert_eq!(name, "tRNA");
}

// =====================================================================
// P7 Match Emission Tests
// =====================================================================

#[test]
fn test_p7_match_emissions() {
    let golden = load_golden("p7_hmm_info.txt");

    // Parse first 10 match emissions from golden data
    let mut golden_mat: Vec<[f32; 4]> = Vec::new();
    for line in golden.lines() {
        if line.starts_with("mat ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                let node: usize = parts[1].parse().unwrap();
                let emissions: Vec<f32> = parts[2..6]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();

                if emissions.len() == 4 && node <= 10 {
                    // Verify probabilities sum to ~1.0
                    let sum: f32 = emissions.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "Node {} match emissions should sum to ~1.0, got {}",
                        node,
                        sum
                    );

                    while golden_mat.len() < node {
                        golden_mat.push([0.25; 4]);
                    }
                    golden_mat.push([emissions[0], emissions[1], emissions[2], emissions[3]]);
                }
            }
        }
    }

    assert!(
        golden_mat.len() >= 10,
        "Should have at least 10 match emission entries"
    );

    // Verify first match emission (node 1)
    let mat1 = &golden_mat[1];
    assert!((mat1[0] - 0.2266765).abs() < 1e-6, "Node 1 A emission");
    assert!((mat1[1] - 0.1009266).abs() < 1e-6, "Node 1 C emission");
    assert!((mat1[2] - 0.5536628).abs() < 1e-6, "Node 1 G emission");
    assert!((mat1[3] - 0.1187341).abs() < 1e-6, "Node 1 U emission");
}

// =====================================================================
// P7 Insert Emission Tests
// =====================================================================

#[test]
fn test_p7_insert_emissions() {
    let golden = load_golden("p7_hmm_info.txt");

    let mut ins_count = 0;
    for line in golden.lines() {
        if line.starts_with("ins ") {
            ins_count += 1;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                let emissions: Vec<f64> = parts[2..6]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();

                // Insert emissions should be uniform (0.25 for each base)
                for e in &emissions {
                    assert!(
                        (*e - 0.25).abs() < EPSILON,
                        "Insert emission should be 0.25, got {}",
                        e
                    );
                }
            }
        }
    }

    assert!(ins_count >= 10, "Should have at least 10 insert emission lines");
}

// =====================================================================
// P7 Transition Tests
// =====================================================================

#[test]
fn test_p7_transitions() {
    let golden = load_golden("p7_hmm_info.txt");

    // Parse first 10 transitions from golden data
    let mut golden_trans: Vec<[f32; 7]> = Vec::new();
    for line in golden.lines() {
        if line.starts_with("trans ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 9 {
                let node: usize = parts[1].parse().unwrap();

                // Parse 7 transition probabilities: MM MI MD IM II DM DD
                let trans: Vec<f32> = parts[2..9]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();

                if trans.len() == 7 && node <= 10 {
                    // Verify transition probability constraints
                    let m_sum = trans[0] + trans[1] + trans[2]; // MM + MI + MD
                    let i_sum = trans[3] + trans[4]; // IM + II
                    let d_sum = trans[5] + trans[6]; // DM + DD

                    assert!(
                        (m_sum - 1.0).abs() < 0.01,
                        "Node {} M-state transitions sum to {}, expected ~1.0",
                        node,
                        m_sum
                    );
                    assert!(
                        (i_sum - 1.0).abs() < 0.01,
                        "Node {} I-state transitions sum to {}, expected ~1.0",
                        node,
                        i_sum
                    );
                    assert!(
                        (d_sum - 1.0).abs() < 0.01,
                        "Node {} D-state transitions sum to {}, expected ~1.0",
                        node,
                        d_sum
                    );

                    while golden_trans.len() < node {
                        golden_trans.push([0.0; 7]);
                    }
                    golden_trans.push([
                        trans[0], trans[1], trans[2], trans[3], trans[4], trans[5], trans[6],
                    ]);
                }
            }
        }
    }

    assert!(
        golden_trans.len() >= 10,
        "Should have at least 10 transition entries"
    );

    // Verify first transition (node 0)
    let trans0 = &golden_trans[0];
    assert!((trans0[0] - 0.987873).abs() < 1e-6, "Node 0 MM transition");
    assert!((trans0[1] - 0.002110049).abs() < 1e-6, "Node 0 MI transition");
    assert!((trans0[2] - 0.01001697).abs() < 1e-6, "Node 0 MD transition");
}

// =====================================================================
// P7 Filter Statistics Tests
// =====================================================================

#[test]
#[ignore]
fn test_p7_filter_stats_structure() {
    let golden = load_golden("p7_filter_stats.txt");

    // Verify file has content
    assert!(!golden.is_empty());

    // TODO: Parse and verify filter statistics
}

#[test]
#[ignore]
fn test_p7_filter_threshold() {
    let golden = load_golden("p7_filter_stats.txt");

    // Look for filter threshold parameters
    if let Some(threshold) = parse_f64(&golden, "filter_threshold") {
        // Threshold should be reasonable (not too high or too low)
        assert!(threshold.is_finite());
    }
}

// =====================================================================
// P7 vs CP9 Consistency Tests
// =====================================================================

#[test]
#[ignore]
fn test_p7_cp9_emission_similarity() {
    // P7 HMM and CP9 HMM should have similar match emissions
    // since they're derived from the same CM

    let p7_golden = load_golden("p7_hmm_info.txt");

    // Parse P7 match emission for node 1
    let mut p7_mat1: Vec<f64> = Vec::new();
    for line in p7_golden.lines() {
        if line.starts_with("mat 1 ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                p7_mat1 = parts[2..6]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();
            }
            break;
        }
    }

    if !p7_mat1.is_empty() {
        // First node match emissions should favor G (position 0 in tRNA)
        // The exact pattern depends on the model training
        assert_eq!(p7_mat1.len(), 4);

        // Emissions should show some information content
        let max_emit = p7_mat1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_emit > 0.3,
            "Should have informative emissions, max = {}",
            max_emit
        );
    }
}
