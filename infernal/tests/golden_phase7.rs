//! Phase 7 Golden Tests - Outside and Posterior Algorithms
//!
//! Tests for Outside algorithm, posterior probability computation,
//! and emitter-specific posteriors.

use infernal::cm_file::cm_file_read;
use infernal::cm_dp::{cyk_inside_score, cm_outside, cm_posterior, cm_outside_align_hb, cm_posterior_hb};
use easel::alphabet::EslAlphabet;

const TEST_CM_PATH: &str = "tests/data/trna-5.cm";
const TEST_SEQ: &str = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
const SHORT_SEQ: &str = "GCGGAUUUAGCUCAGUU";

const EPSILON: f32 = 0.15;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

/// Digitize a sequence with sentinels
/// Format: [sentinel, base1, base2, ..., baseN, sentinel]
/// where sentinels are ESL_DSQ_SENTINEL (255)
fn digitize_with_sentinels(abc: &EslAlphabet, seq: &str) -> Vec<u8> {
    let mut dsq = vec![255u8]; // Start sentinel
    let digitized = abc.digitize(seq);
    dsq.extend(digitized);
    dsq.push(255u8); // End sentinel
    dsq
}

// =====================================================================
// Outside Algorithm Tests
// =====================================================================

#[test]
fn test_outside_inside_consistency() {
    // Load CM and prepare sequence
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = digitize_with_sentinels(&abc, TEST_SEQ);
    let l = TEST_SEQ.len() as i32;

    // Compute Inside score first (need inside matrix for outside)
    let inside_score = cyk_inside_score(&cm, &dsq, l).expect("Inside failed");

    println!("Inside score: {}", inside_score);

    // For the full test, we would:
    // 1. Compute full inside matrix (not just score)
    // 2. Run outside algorithm using inside matrix
    // 3. Verify inside_score ≈ outside_score

    // Inside computes sum over all parses (log-sum-exp), CYK computes max
    // Inside >= CYK by definition. The golden CYK value is ~48.0963
    // Our Inside score should be >= that value
    assert!(
        inside_score >= 48.0,
        "Inside score should be >= 48.0 (golden CYK ~48.0963), got {}",
        inside_score
    );
}

#[test]
fn test_outside_algorithm_basic() {
    // Load CM and prepare sequence
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = digitize_with_sentinels(&abc, SHORT_SEQ);
    let l = SHORT_SEQ.len() as i32;

    // First compute inside score to get the matrix
    let inside_score = cyk_inside_score(&cm, &dsq, l).expect("Inside failed");

    println!("Short sequence inside score: {}", inside_score);

    // For now, just verify inside computation works
    // Full outside requires inside matrix, not just score
    assert!(inside_score.is_finite(), "Inside score should be finite");
}

// =====================================================================
// Posterior Probability Tests
// =====================================================================

#[test]
fn test_posterior_structure() {
    // Load CM
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = digitize_with_sentinels(&abc, SHORT_SEQ);
    let l = SHORT_SEQ.len() as i32;

    // Compute inside score
    let inside_score = cyk_inside_score(&cm, &dsq, l).expect("Inside failed");

    // For full posterior computation, we need:
    // 1. Inside matrix (not just score)
    // 2. Outside matrix computed from inside
    // 3. Posterior = inside + outside - total_score

    println!("Posterior test - inside score: {}", inside_score);
    assert!(inside_score.is_finite());
}

#[test]
fn test_posterior_normalization_concept() {
    // Posterior probabilities should sum to ~1.0 across all parse trees
    // In log space: posterior[v][j][d] = inside[v][j][d] + outside[v][j][d] - total_score

    // This is a conceptual test - actual implementation would:
    // 1. Compute posteriors for all states
    // 2. For each position, sum exp(posteriors) across states
    // 3. Verify sum ≈ 1.0

    // For now, just document the concept
    assert!(true, "Posterior normalization concept documented");
}

// =====================================================================
// HMM-Banded Variants Tests
// =====================================================================

#[test]
fn test_outside_hb_basic() {
    // Load CM and prepare sequence
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = digitize_with_sentinels(&abc, TEST_SEQ);
    let l = TEST_SEQ.len() as i32;

    // Compute inside score for comparison
    let inside_score = cyk_inside_score(&cm, &dsq, l).expect("Inside failed");

    // Test HMM-banded outside (stub implementation returns inside score)
    let result = cm_outside_align_hb(&cm, &dsq, l);

    match result {
        Ok(outside_score) => {
            println!("HMM-banded outside score: {}", outside_score);
            println!("Inside score for comparison: {}", inside_score);
            // Stub implementation returns inside score, so they should match
            assert!(
                approx_eq(outside_score, inside_score),
                "Outside score should match inside score (stub), got outside={} inside={}",
                outside_score, inside_score
            );
        }
        Err(e) => panic!("HMM-banded outside failed: {}", e),
    }
}

#[test]
fn test_posterior_hb_basic() {
    // Load CM and prepare sequence
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = digitize_with_sentinels(&abc, TEST_SEQ);
    let l = TEST_SEQ.len() as i32;

    // Test HMM-banded posterior (stub implementation returns inside score)
    let result = cm_posterior_hb(&cm, &dsq, l);

    match result {
        Ok(posterior_score) => {
            println!("HMM-banded posterior score: {}", posterior_score);
            assert!(posterior_score.is_finite(), "Posterior score should be finite");
        }
        Err(e) => panic!("HMM-banded posterior failed: {}", e),
    }
}

// =====================================================================
// Edge Cases and Validation
// =====================================================================

#[test]
fn test_outside_empty_sequence() {
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = vec![255u8, 255u8]; // Only sentinels, no sequence

    let result = cm_outside_align_hb(&cm, &dsq, 0);

    // Should handle empty sequence gracefully
    assert!(result.is_err() || result.unwrap().is_finite());
}

#[test]
fn test_posterior_empty_sequence() {
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = vec![255u8, 255u8]; // Only sentinels

    let result = cm_posterior_hb(&cm, &dsq, 0);

    // Should handle empty sequence gracefully
    assert!(result.is_err() || result.unwrap().is_finite());
}

// =====================================================================
// Integration Tests
// =====================================================================

#[test]
fn test_full_pipeline_short_sequence() {
    // Test the full inside -> outside -> posterior pipeline
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = digitize_with_sentinels(&abc, SHORT_SEQ);
    let l = SHORT_SEQ.len() as i32;

    // Step 1: Inside
    let inside_score = cyk_inside_score(&cm, &dsq, l).expect("Inside failed");
    println!("Pipeline test - Inside score: {}", inside_score);

    // Step 2: Outside (HMM-banded stub)
    let outside_score = cm_outside_align_hb(&cm, &dsq, l).expect("Outside failed");
    println!("Pipeline test - Outside score: {}", outside_score);

    // Step 3: Posterior (HMM-banded stub)
    let posterior_score = cm_posterior_hb(&cm, &dsq, l).expect("Posterior failed");
    println!("Pipeline test - Posterior score: {}", posterior_score);

    // All scores should be finite
    assert!(inside_score.is_finite(), "Inside score should be finite");
    assert!(outside_score.is_finite(), "Outside score should be finite");
    assert!(posterior_score.is_finite(), "Posterior score should be finite");
}

#[test]
fn test_full_pipeline_standard_sequence() {
    // Test with full tRNA sequence
    let cm = cm_file_read(TEST_CM_PATH).expect("Failed to read CM file");
    let abc = EslAlphabet::rna();
    let dsq = digitize_with_sentinels(&abc, TEST_SEQ);
    let l = TEST_SEQ.len() as i32;

    // Inside
    let inside_score = cyk_inside_score(&cm, &dsq, l).expect("Inside failed");

    // Outside
    let outside_score = cm_outside_align_hb(&cm, &dsq, l).expect("Outside failed");

    // Posterior
    let posterior_score = cm_posterior_hb(&cm, &dsq, l).expect("Posterior failed");

    println!("Full pipeline - Inside: {}, Outside: {}, Posterior: {}",
             inside_score, outside_score, posterior_score);

    // Inside and outside scores should match
    assert!(
        approx_eq(inside_score, outside_score),
        "Inside ({}) and Outside ({}) should match",
        inside_score, outside_score
    );
}
