//! Phase 3 Golden Tests - Math Functions
//!
//! Tests for mathematical functions including Gumbel distribution
//! and random number generation.
//! Golden data generated from Easel math modules.

use crate::gumbel::{esl_gumbel_logpdf, esl_gumbel_logsurv, esl_gumbel_pdf, esl_gumbel_surv};
use crate::random::EslRandom;
use std::fs;

fn load_golden(filename: &str) -> String {
    fs::read_to_string(format!("../tests/golden/phase3/{}", filename))
        .expect("Failed to load golden file")
}

const EPSILON: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.abs() < EPSILON && b.abs() < EPSILON {
        return true;
    }
    (a - b).abs() / (a.abs().max(b.abs()).max(1.0)) < EPSILON
}

// =====================================================================
// Gumbel Distribution Tests
// =====================================================================

/// Parse Gumbel test values from golden file
fn parse_gumbel_values(content: &str) -> Vec<(f64, f64, f64, f64, f64)> {
    let mut results = Vec::new();
    let mut current_x: Option<f64> = None;
    let mut logsurv: Option<f64> = None;
    let mut surv: Option<f64> = None;
    let mut pdf: Option<f64> = None;
    let mut logpdf: Option<f64> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("x=") {
            // Save previous record if complete
            if let (Some(x), Some(ls), Some(s), Some(p), Some(lp)) =
                (current_x, logsurv, surv, pdf, logpdf)
            {
                results.push((x, ls, s, p, lp));
            }
            current_x = trimmed.split('=').nth(1).and_then(|s| s.parse().ok());
            logsurv = None;
            surv = None;
            pdf = None;
            logpdf = None;
        } else if trimmed.starts_with("logsurv=") {
            logsurv = trimmed.split('=').nth(1).and_then(|s| s.parse().ok());
        } else if trimmed.starts_with("surv=") {
            surv = trimmed.split('=').nth(1).and_then(|s| s.parse().ok());
        } else if trimmed.starts_with("pdf=") {
            pdf = trimmed.split('=').nth(1).and_then(|s| s.parse().ok());
        } else if trimmed.starts_with("logpdf=") {
            logpdf = trimmed.split('=').nth(1).and_then(|s| s.parse().ok());
        }
    }

    // Save final record
    if let (Some(x), Some(ls), Some(s), Some(p), Some(lp)) =
        (current_x, logsurv, surv, pdf, logpdf)
    {
        results.push((x, ls, s, p, lp));
    }

    results
}

#[test]
fn test_gumbel_logsurv() {
    let golden = load_golden("gumbel_values.txt");
    let values = parse_gumbel_values(&golden);

    assert!(!values.is_empty(), "Should have Gumbel test values");

    // Test with mu=0, lambda=1 (standard Gumbel)
    for (x, expected_logsurv, _, _, _) in &values {
        let actual = esl_gumbel_logsurv(*x, 0.0, 1.0);
        assert!(
            approx_eq(actual, *expected_logsurv),
            "logsurv mismatch at x={}: actual={}, expected={}",
            x,
            actual,
            expected_logsurv
        );
    }
}

#[test]
fn test_gumbel_surv() {
    let golden = load_golden("gumbel_values.txt");
    let values = parse_gumbel_values(&golden);

    for (x, _, expected_surv, _, _) in &values {
        let actual = esl_gumbel_surv(*x, 0.0, 1.0);
        assert!(
            approx_eq(actual, *expected_surv),
            "surv mismatch at x={}: actual={}, expected={}",
            x,
            actual,
            expected_surv
        );
    }
}

#[test]
fn test_gumbel_pdf() {
    let golden = load_golden("gumbel_values.txt");
    let values = parse_gumbel_values(&golden);

    for (x, _, _, expected_pdf, _) in &values {
        let actual = esl_gumbel_pdf(*x, 0.0, 1.0);
        assert!(
            approx_eq(actual, *expected_pdf),
            "pdf mismatch at x={}: actual={}, expected={}",
            x,
            actual,
            expected_pdf
        );
    }
}

#[test]
fn test_gumbel_logpdf() {
    let golden = load_golden("gumbel_values.txt");
    let values = parse_gumbel_values(&golden);

    for (x, _, _, _, expected_logpdf) in &values {
        let actual = esl_gumbel_logpdf(*x, 0.0, 1.0);
        assert!(
            approx_eq(actual, *expected_logpdf),
            "logpdf mismatch at x={}: actual={}, expected={}",
            x,
            actual,
            expected_logpdf
        );
    }
}

#[test]
fn test_gumbel_extreme_values() {
    let golden = load_golden("gumbel_values.txt");
    let values = parse_gumbel_values(&golden);

    // Find x=10 test case
    for (x, logsurv, surv, _, _) in &values {
        if (*x - 10.0).abs() < 0.01 {
            let actual_surv = esl_gumbel_surv(*x, 0.0, 1.0);
            let actual_logsurv = esl_gumbel_logsurv(*x, 0.0, 1.0);

            assert!(
                approx_eq(actual_surv, *surv),
                "surv mismatch at x=10: actual={}, expected={}",
                actual_surv,
                surv
            );
            assert!(
                approx_eq(actual_logsurv, *logsurv),
                "logsurv mismatch at x=10: actual={}, expected={}",
                actual_logsurv,
                logsurv
            );
            return;
        }
    }
    panic!("x=10 test case not found in golden file");
}

// =====================================================================
// Random Sequence Tests
// =====================================================================

fn parse_random_sequence(content: &str) -> Vec<f64> {
    content
        .lines()
        .filter(|line| !line.trim().starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.trim().parse().ok())
        .collect()
}

#[test]
fn test_random_sequence_reproducibility() {
    let golden = load_golden("random_sequence.txt");
    let expected: Vec<f64> = parse_random_sequence(&golden);

    assert!(!expected.is_empty(), "Should have random values");

    // Generate sequence with seed=42
    let mut rng = EslRandom::new(42);
    let actual: Vec<f64> = (0..expected.len()).map(|_| rng.random()).collect();

    // Verify first 5 values match golden file
    assert!(
        approx_eq(actual[0], expected[0]),
        "First value mismatch: {} vs {}",
        actual[0],
        expected[0]
    );
    assert!(
        approx_eq(actual[1], expected[1]),
        "Second value mismatch: {} vs {}",
        actual[1],
        expected[1]
    );
    assert!(
        approx_eq(actual[2], expected[2]),
        "Third value mismatch: {} vs {}",
        actual[2],
        expected[2]
    );
    assert!(
        approx_eq(actual[3], expected[3]),
        "Fourth value mismatch: {} vs {}",
        actual[3],
        expected[3]
    );
    assert!(
        approx_eq(actual[4], expected[4]),
        "Fifth value mismatch: {} vs {}",
        actual[4],
        expected[4]
    );

    // Verify all 100 values
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e),
            "Mismatch at index {}: actual={}, expected={}",
            i,
            a,
            e
        );
    }
}

#[test]
fn test_random_sequence_distribution() {
    let golden = load_golden("random_sequence.txt");
    let values: Vec<f64> = parse_random_sequence(&golden);

    // All values should be in [0, 1)
    for (i, v) in values.iter().enumerate() {
        assert!(
            *v >= 0.0 && *v < 1.0,
            "Value at index {} out of range: {}",
            i,
            v
        );
    }

    // Verify mean is approximately 0.5 (within reasonable bounds for 100 samples)
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    assert!(
        mean > 0.3 && mean < 0.7,
        "Mean {} is outside expected range for uniform distribution",
        mean
    );
}

// =====================================================================
// Random Integer Tests
// =====================================================================

fn parse_random_ints(content: &str) -> Vec<u32> {
    content
        .lines()
        .filter(|line| !line.trim().starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.trim().parse().ok())
        .collect()
}

#[test]
fn test_random_ints() {
    let golden = load_golden("random_ints.txt");
    let expected: Vec<u32> = parse_random_ints(&golden);

    assert!(!expected.is_empty(), "Should have random integers");

    // Generate sequence with seed=42
    let mut rng = EslRandom::new(42);
    let actual: Vec<u32> = (0..expected.len()).map(|_| rng.random_int(10)).collect();

    // Verify first 10 values
    for i in 0..10.min(expected.len()) {
        assert_eq!(
            actual[i], expected[i],
            "Integer mismatch at index {}: actual={}, expected={}",
            i, actual[i], expected[i]
        );
    }

    // Verify all values
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a, e,
            "Integer mismatch at index {}: actual={}, expected={}",
            i, a, e
        );
    }
}

#[test]
fn test_random_int_bounds() {
    let golden = load_golden("random_ints.txt");
    let values: Vec<u32> = parse_random_ints(&golden);

    // All values should be in [0, 10)
    for (i, v) in values.iter().enumerate() {
        assert!(
            *v < 10,
            "Integer at index {} out of range [0,10): {}",
            i,
            v
        );
    }
}

// =====================================================================
// Additional Tests
// =====================================================================

#[test]
fn test_rng_determinism() {
    // Same seed should produce identical sequences
    let mut rng1 = EslRandom::new(42);
    let mut rng2 = EslRandom::new(42);

    for i in 0..1000 {
        let v1 = rng1.random();
        let v2 = rng2.random();
        assert!(
            approx_eq(v1, v2),
            "Sequence diverged at iteration {}: {} vs {}",
            i,
            v1,
            v2
        );
    }
}

#[test]
fn test_different_seeds_different_sequences() {
    let mut rng1 = EslRandom::new(42);
    let mut rng2 = EslRandom::new(43);

    // Different seeds should produce different sequences
    let v1: Vec<f64> = (0..10).map(|_| rng1.random()).collect();
    let v2: Vec<f64> = (0..10).map(|_| rng2.random()).collect();

    // At least one value should differ
    assert!(
        v1.iter().zip(v2.iter()).any(|(a, b)| !approx_eq(*a, *b)),
        "Different seeds produced identical sequences"
    );
}
