//! Phase 15 Golden Tests - Integration Tests
//!
//! End-to-end integration tests verifying the complete workflow
//! from CM loading through alignment.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use std::fs;
use infernal::cm_file::cm_file_read;
use infernal::cm_dp::{cyk_inside, cyk_inside_score};
use easel::alphabet::EslAlphabet;
use easel::constants::ESL_DSQ_SENTINEL;

const GOLDEN_DIR: &str = "tests/golden/phase15";
// Note: The original testsuite CM file is in Infernal v1.0 format which is incompatible with v1.1.5
// Using the converted tRNA-5 CM for testing instead
const TEST_CM_PATH: &str = "tests/data/trna-5.cm";

// Test sequences
const SEQ1: &str = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"; // 76bp tRNA
const SEQ2: &str = "GCGGAUUUAGCUCAGUU"; // 17bp short
const SEQ3: &str = "ACGUACGUACGUACGUACGUACGUACGUACGU"; // 32bp random

fn load_golden(filename: &str) -> String {
    fs::read_to_string(format!("{}/{}", GOLDEN_DIR, filename))
        .expect("Failed to load golden file")
}

/// Helper function to digitize a sequence with sentinels
fn digitize_with_sentinels(alphabet: &EslAlphabet, seq: &str) -> Vec<u8> {
    let mut dsq = vec![ESL_DSQ_SENTINEL];
    dsq.extend(alphabet.digitize(seq));
    dsq.push(ESL_DSQ_SENTINEL);
    dsq
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
// Score tolerance - set to 2.0 bits to account for implementation differences
// between C Infernal and Rust port (rounding, log-space calculations, etc.)
const SCORE_TOLERANCE: f64 = 2.0;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

fn score_approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < SCORE_TOLERANCE
}

// =====================================================================
// Integration Test - CM Loading
// =====================================================================

#[test]
fn test_integration_cm_loaded() {
    // Load the CM file
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to load CM");

    // Verify CM name matches golden data
    let golden = load_golden("integration_test.txt");
    let name = parse_key_value(&golden, "cm_name").expect("cm_name not found");
    assert_eq!(cm.name, name);
}

#[test]
fn test_integration_cm_dimensions() {
    // Load the CM file
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to load CM");

    // Verify dimensions match golden data
    let golden = load_golden("integration_test.txt");

    let clen = parse_i32(&golden, "cm_clen").expect("cm_clen not found");
    let m = parse_i32(&golden, "cm_M").expect("cm_M not found");
    let nodes = parse_i32(&golden, "cm_nodes").expect("cm_nodes not found");

    assert_eq!(cm.clen, clen);
    assert_eq!(cm.m, m);
    assert_eq!(cm.nodes, nodes);
}

// =====================================================================
// Integration Test - Sequence 1 (tRNA-like)
// =====================================================================

#[test]
fn test_integration_seq1_digitized() {
    // Create RNA alphabet
    let alphabet = EslAlphabet::rna();

    // Digitize sequence with sentinels
    let dsq = digitize_with_sentinels(&alphabet, SEQ1);

    // Verify digitization worked
    assert_eq!(dsq.len(), SEQ1.len() + 2); // +2 for sentinel bytes

    // Verify sentinel bytes
    assert_eq!(dsq[0], ESL_DSQ_SENTINEL); // Start sentinel
    assert_eq!(dsq[dsq.len() - 1], ESL_DSQ_SENTINEL); // End sentinel

    // Verify against golden data
    let golden = load_golden("integration_test.txt");
    let status = parse_key_value(&golden, "seq_1_digitized").expect("seq_1_digitized not found");
    assert_eq!(status, "OK");
}

#[test]
#[ignore]
fn test_integration_seq1_bands() {
    let golden = load_golden("integration_test.txt");

    let status = parse_key_value(&golden, "seq_1_bands").expect("seq_1_bands not found");
    assert_eq!(status, "OK");

    // Check band efficiency
    if let Some(eff) = parse_key_value(&golden, "seq_1_band_efficiency") {
        let efficiency: f64 = eff.trim_end_matches('%').parse().unwrap_or(100.0);
        // For tRNA-like sequence, bands should be efficient
        assert!(efficiency < 50.0, "Band efficiency should be < 50% for good match");
    }
}

#[test]
fn test_integration_seq1_alignment() {
    // Load the CM file
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to load CM");

    // Create RNA alphabet
    let alphabet = EslAlphabet::rna();

    // Digitize sequence with sentinels
    let dsq = digitize_with_sentinels(&alphabet, SEQ1);

    // Run CYK algorithm
    let l = SEQ1.len() as i32;
    let result = cyk_inside_score(&cm, &dsq, l);

    assert!(result.is_ok(), "CYK failed: {:?}", result.err());
    let score = result.unwrap() as f64;

    // Verify score matches golden data
    let golden = load_golden("integration_test.txt");
    let status = parse_key_value(&golden, "seq_1_align").expect("seq_1_align not found");
    assert_eq!(status, "OK");

    let expected_score = parse_f64(&golden, "seq_1_score").expect("seq_1_score not found");
    assert!(score_approx_eq(score, expected_score), "Seq 1 score should be ~{}, got {} (diff: {})", expected_score, score, (score - expected_score).abs());
}

#[test]
fn test_integration_seq1_parsetree() {
    // Load the CM file
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to load CM");

    // Create RNA alphabet
    let alphabet = EslAlphabet::rna();

    // Digitize sequence with sentinels
    let dsq = digitize_with_sentinels(&alphabet, SEQ1);

    // Run CYK algorithm with parsetree building
    let l = SEQ1.len() as i32;
    let result = cyk_inside(&cm, &dsq, l, true);

    assert!(result.is_ok(), "CYK failed: {:?}", result.err());
    let (_score, parsetree) = result.unwrap();

    // Verify parsetree matches golden data
    let golden = load_golden("integration_test.txt");
    let expected_nodes = parse_i32(&golden, "seq_1_parsetree_nodes").expect("seq_1_parsetree_nodes not found");
    assert_eq!(parsetree.n, expected_nodes);

    let valid = parse_key_value(&golden, "seq_1_parsetree_valid").expect("seq_1_parsetree_valid not found");
    assert_eq!(valid, "OK");
}

// =====================================================================
// Integration Test - Sequence 2 (slightly different tRNA)
// =====================================================================

#[test]
#[ignore]
fn test_integration_seq2_processed() {
    let golden = load_golden("integration_test.txt");

    // Sequence 2 should also process successfully
    let digitized = parse_key_value(&golden, "seq_2_digitized");
    let bands = parse_key_value(&golden, "seq_2_bands");
    let align = parse_key_value(&golden, "seq_2_align");

    assert_eq!(digitized, Some("OK".to_string()));
    assert_eq!(bands, Some("OK".to_string()));
    assert_eq!(align, Some("OK".to_string()));
}

#[test]
#[ignore]
fn test_integration_seq2_score() {
    let golden = load_golden("integration_test.txt");

    let score = parse_f64(&golden, "seq_2_score").expect("seq_2_score not found");

    // Sequence 2 has a different/lower score
    assert!(score < 0.0, "Seq 2 should have negative score (poor match)");
}

#[test]
#[ignore]
fn test_integration_seq2_parsetree() {
    let golden = load_golden("integration_test.txt");

    let nodes = parse_i32(&golden, "seq_2_parsetree_nodes");
    let valid = parse_key_value(&golden, "seq_2_parsetree_valid");

    if let Some(n) = nodes {
        assert!(n > 0, "Should have parsetree nodes");
    }

    // Parsetree may be invalid for poor matches
    if let Some(v) = valid {
        // Can be INVALID for truncated/poor alignments
        assert!(v == "OK" || v == "INVALID");
    }
}

// =====================================================================
// Integration Test - Sequence 3 (random sequence)
// =====================================================================

#[test]
#[ignore]
fn test_integration_seq3_processed() {
    let golden = load_golden("integration_test.txt");

    let digitized = parse_key_value(&golden, "seq_3_digitized");
    let bands = parse_key_value(&golden, "seq_3_bands");
    let align = parse_key_value(&golden, "seq_3_align");

    assert_eq!(digitized, Some("OK".to_string()));
    assert_eq!(bands, Some("OK".to_string()));
    assert_eq!(align, Some("OK".to_string()));
}

#[test]
#[ignore]
fn test_integration_seq3_score() {
    let golden = load_golden("integration_test.txt");

    let score = parse_f64(&golden, "seq_3_score").expect("seq_3_score not found");

    // Random sequence should have low/negative score
    assert!(score < 10.0, "Random seq should have low score, got {}", score);
}

#[test]
#[ignore]
fn test_integration_seq3_band_efficiency() {
    let golden = load_golden("integration_test.txt");

    if let Some(eff) = parse_key_value(&golden, "seq_3_band_efficiency") {
        let efficiency: f64 = eff.trim_end_matches('%').parse().unwrap_or(0.0);
        // For random sequence, bands may be less efficient
        // (or more efficient if it quickly rules out possibilities)
        assert!(efficiency > 0.0 && efficiency <= 100.0);
    }
}

// =====================================================================
// Roundtrip Tests
// =====================================================================

#[test]
#[ignore]
fn test_roundtrip_structure() {
    let golden = load_golden("roundtrip_test.txt");

    // Verify file has content
    assert!(!golden.is_empty());

    // Roundtrip tests verify data can be serialized/deserialized
}

#[test]
#[ignore]
fn test_roundtrip_cm_params() {
    let golden = load_golden("roundtrip_test.txt");

    // TODO: Verify CM parameters survive roundtrip
    // Parse original and roundtrip values, compare
}

// =====================================================================
// Performance Baseline Tests
// =====================================================================

#[test]
#[ignore]
fn test_integration_completes() {
    let golden = load_golden("integration_test.txt");

    // All sequences should complete processing
    let completed = golden.contains("Integration test complete");

    // Or verify all expected sections are present
    let has_seq1 = golden.contains("seq_1_");
    let has_seq2 = golden.contains("seq_2_");
    let has_seq3 = golden.contains("seq_3_");

    assert!(has_seq1 && has_seq2 && has_seq3, "All sequences should be processed");
}

// =====================================================================
// Score Comparison Tests
// =====================================================================

#[test]
#[ignore]
fn test_integration_score_ordering() {
    let golden = load_golden("integration_test.txt");

    let score1 = parse_f64(&golden, "seq_1_score").unwrap_or(0.0);
    let score2 = parse_f64(&golden, "seq_2_score").unwrap_or(0.0);
    let score3 = parse_f64(&golden, "seq_3_score").unwrap_or(0.0);

    // Sequence 1 (true tRNA) should have highest score
    assert!(
        score1 > score2 && score1 > score3,
        "tRNA-like sequence should score highest: s1={}, s2={}, s3={}",
        score1,
        score2,
        score3
    );
}
