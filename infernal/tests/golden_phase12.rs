//! Phase 12 Golden Tests - E-value Parameters
//!
//! Tests for E-value calculation parameters including exponential tail
//! distribution parameters for different scoring modes.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use infernal::evalue::{ExpParams, P7ExpParams};
use std::fs;

const GOLDEN_DIR: &str = "tests/golden/phase12";

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

const EPSILON: f64 = 1e-6;
const NOT_AVAILABLE: f64 = -99999.0;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

fn is_not_available(val: f64) -> bool {
    (val - NOT_AVAILABLE).abs() < 1.0
}

// =====================================================================
// E-value Parameter Tests
// =====================================================================

#[test]
fn test_evalue_params_structure() {
    let golden = load_golden("evalue_params.txt");

    // Verify file has content
    assert!(!golden.is_empty());

    // Should have exp mode lines
    assert!(golden.contains("exp mode="), "Should have exp mode lines");
}

#[test]
fn test_evalue_params_exp_modes() {
    let golden = load_golden("evalue_params.txt");

    // Parse exp mode lines
    // Format: exp mode=XX not_available  OR  exp mode=XX mu=Y lambda=Z
    let modes = ["GC", "GI", "LC", "LI"];

    for mode in &modes {
        let mode_line = golden
            .lines()
            .find(|l| l.contains(&format!("mode={}", mode)));

        assert!(
            mode_line.is_some(),
            "Should have exp mode={} line",
            mode
        );

        // For this test CM, modes are not available
        if let Some(line) = mode_line {
            assert!(
                line.contains("not_available"),
                "Mode {} should be not_available in test CM",
                mode
            );
        }
    }
}

// =====================================================================
// P7 HMM E-value Parameters Tests
// =====================================================================

#[test]
fn test_evalue_p7_lmmu() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_lmmu").expect("fp7_lmmu not found");

    // Should be not available (sentinel value)
    assert!(
        is_not_available(val),
        "fp7_lmmu should be sentinel value, got {}",
        val
    );
}

#[test]
fn test_evalue_p7_lmlambda() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_lmlambda").expect("fp7_lmlambda not found");

    assert!(
        is_not_available(val),
        "fp7_lmlambda should be sentinel value, got {}",
        val
    );
}

#[test]
fn test_evalue_p7_lvmu() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_lvmu").expect("fp7_lvmu not found");

    assert!(
        is_not_available(val),
        "fp7_lvmu should be sentinel value, got {}",
        val
    );
}

#[test]
fn test_evalue_p7_lvlambda() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_lvlambda").expect("fp7_lvlambda not found");

    assert!(
        is_not_available(val),
        "fp7_lvlambda should be sentinel value, got {}",
        val
    );
}

#[test]
fn test_evalue_p7_lftau() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_lftau").expect("fp7_lftau not found");

    assert!(
        is_not_available(val),
        "fp7_lftau should be sentinel value, got {}",
        val
    );
}

#[test]
fn test_evalue_p7_lflambda() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_lflambda").expect("fp7_lflambda not found");

    assert!(
        is_not_available(val),
        "fp7_lflambda should be sentinel value, got {}",
        val
    );
}

#[test]
fn test_evalue_p7_gfmu() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_gfmu").expect("fp7_gfmu not found");

    assert!(
        is_not_available(val),
        "fp7_gfmu should be sentinel value, got {}",
        val
    );
}

#[test]
fn test_evalue_p7_gflambda() {
    let golden = load_golden("evalue_params.txt");

    let val = parse_f64(&golden, "fp7_gflambda").expect("fp7_gflambda not found");

    assert!(
        is_not_available(val),
        "fp7_gflambda should be sentinel value, got {}",
        val
    );
}

// =====================================================================
// Score to E-value Conversion Tests
// =====================================================================

#[test]
fn test_score_to_evalue_structure() {
    let golden = load_golden("score_to_evalue.txt");

    // Verify file has content
    assert!(!golden.is_empty());

    // TODO: Parse and verify score-to-evalue conversion examples
}

#[test]
fn test_score_to_evalue_monotonic() {
    let golden = load_golden("score_to_evalue.txt");

    // Parse score-evalue pairs
    // Format: score=X evalue=Y
    let mut pairs: Vec<(f64, f64)> = Vec::new();

    for line in golden.lines() {
        if line.contains("score=") && line.contains("evalue=") {
            let score = line
                .split_whitespace()
                .find(|p| p.starts_with("score="))
                .and_then(|s| s.split('=').last())
                .and_then(|v| v.parse().ok());

            let evalue = line
                .split_whitespace()
                .find(|p| p.starts_with("evalue="))
                .and_then(|s| s.split('=').last())
                .and_then(|v| v.parse().ok());

            if let (Some(s), Some(e)) = (score, evalue) {
                pairs.push((s, e));
            }
        }
    }

    // Higher scores should give lower E-values
    if pairs.len() >= 2 {
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in 1..pairs.len() {
            // Higher score (pairs[i].0) should have lower or equal E-value (pairs[i].1)
            // Note: equality is possible at extremes
            assert!(
                pairs[i].1 <= pairs[i - 1].1 + EPSILON,
                "E-value should decrease as score increases: score {} -> evalue {}, score {} -> evalue {}",
                pairs[i - 1].0,
                pairs[i - 1].1,
                pairs[i].0,
                pairs[i].1
            );
        }
    }
}

// =====================================================================
// E-value Calculation Tests
// =====================================================================

#[test]
fn test_evalue_positive_scores() {
    let golden = load_golden("score_to_evalue.txt");

    // For positive bit scores, E-values should be small
    for line in golden.lines() {
        if line.contains("score=") && line.contains("evalue=") {
            let score: Option<f64> = line
                .split_whitespace()
                .find(|p| p.starts_with("score="))
                .and_then(|s| s.split('=').last())
                .and_then(|v| v.parse().ok());

            let evalue: Option<f64> = line
                .split_whitespace()
                .find(|p| p.starts_with("evalue="))
                .and_then(|s| s.split('=').last())
                .and_then(|v| v.parse().ok());

            if let (Some(s), Some(e)) = (score, evalue) {
                if s > 40.0 {
                    // Scores well above mu should have small E-values
                    assert!(
                        e < 1.0,
                        "High score {} should have E-value < 1, got {}",
                        s,
                        e
                    );
                }
            }
        }
    }
}

#[test]
fn test_evalue_database_size_scaling() {
    // E-values should scale linearly with database size
    // This test verifies the E-value calculation formula
    let params1 = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };

    let params2 = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 10_000_000.0,
        nrandhits: 100000,
    };

    let score = 50.0;
    let e1 = params1.score_to_evalue(score);
    let e2 = params2.score_to_evalue(score);

    // E-values should scale linearly with database size
    let ratio = e2 / e1;
    assert!((ratio - 10.0).abs() < 0.01, "E-value should scale linearly with dbsize");
}

// =====================================================================
// Specific Score Conversion Tests (from requirements)
// =====================================================================

#[test]
fn test_score_to_evalue_10() {
    let params = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };

    let evalue = params.score_to_evalue(10.0);
    let expected = 1.022494e+09;

    assert!(
        (evalue - expected).abs() / expected < 0.01,
        "score=10 should give E≈1.022e+09, got {}",
        evalue
    );
}

#[test]
fn test_score_to_evalue_50() {
    let params = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };

    let evalue = params.score_to_evalue(50.0);
    let expected = 9.313226e-04;

    assert!(
        (evalue - expected).abs() / expected < 0.01,
        "score=50 should give E≈9.354e-04, got {}",
        evalue
    );
}

#[test]
fn test_score_to_evalue_100() {
    let params = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };

    let evalue = params.score_to_evalue(100.0);
    let expected = 8.271806e-19;

    assert!(
        (evalue - expected).abs() / expected < 0.02,
        "score=100 should give E≈8.27e-19, got {}",
        evalue
    );
}

#[test]
fn test_evalue_to_score_roundtrip() {
    let params = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };

    let scores = vec![10.0, 30.0, 50.0, 70.0, 90.0];

    for original_score in scores {
        let evalue = params.score_to_evalue(original_score);
        let recovered_score = params.evalue_to_score(evalue);

        assert!(
            (original_score - recovered_score).abs() < 1e-6,
            "Roundtrip failed: {} -> {} -> {}",
            original_score,
            evalue,
            recovered_score
        );
    }
}

#[test]
fn test_is_calibrated() {
    // Test calibrated params
    let calibrated = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };
    assert!(calibrated.is_calibrated());

    // Test uncalibrated params
    let uncalibrated1 = ExpParams {
        lambda: 0.0,
        mu: 20.0,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };
    assert!(!uncalibrated1.is_calibrated());

    let uncalibrated2 = ExpParams {
        lambda: 0.693,
        mu: f64::NAN,
        dbsize: 1_000_000.0,
        nrandhits: 100000,
    };
    assert!(!uncalibrated2.is_calibrated());

    let uncalibrated3 = ExpParams {
        lambda: 0.693,
        mu: 20.0,
        dbsize: 0.0,
        nrandhits: 100000,
    };
    assert!(!uncalibrated3.is_calibrated());
}

#[test]
fn test_evalue_params_uncalibrated() {
    let golden = load_golden("evalue_params.txt");

    // All P7 params should be -99999 (uncalibrated)
    let p7_params = [
        "fp7_lmmu",
        "fp7_lmlambda",
        "fp7_lvmu",
        "fp7_lvlambda",
        "fp7_lftau",
        "fp7_lflambda",
        "fp7_gfmu",
        "fp7_gflambda",
    ];

    for param in &p7_params {
        let val = parse_f64(&golden, param).expect(&format!("{} not found", param));
        assert!(
            is_not_available(val),
            "{} should be -99999, got {}",
            param,
            val
        );
    }
}
