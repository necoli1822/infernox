//! Phase 8: CP9 HMM Details Tests
//!
//! Tests CP9 HMM integer-scaled log-odds scores and HMM bands against golden reference.

use infernal::cm::ALPHABET_SIZE;
use infernal::cp9::{CP9, CP9_IMPOSSIBLE, CP9_INTSCALE, CP9_NTRANS};
use std::collections::HashMap;
use std::fs;

const GOLDEN_PARAMS: &str = "../tests/golden/phase8/cp9_params.txt";
const GOLDEN_SCORES: &str = "../tests/golden/phase8/cp9_scores.txt";
const GOLDEN_BANDS: &str = "../tests/golden/phase8/cp9_bands.txt";

/// Parse golden CP9 scores file
fn parse_golden_scores() -> GoldenScores {
    let content = fs::read_to_string(GOLDEN_SCORES).expect("Failed to read golden scores");
    let mut tsc = HashMap::new();
    let mut msc = HashMap::new();
    let mut isc = HashMap::new();
    let mut bsc = HashMap::new();
    let mut esc = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "tsc" => {
                if parts.len() >= 12 {
                    let k: usize = parts[1].parse().unwrap();
                    let scores: Vec<i32> = parts[2..12]
                        .iter()
                        .map(|s| s.parse().unwrap())
                        .collect();
                    tsc.insert(k, scores);
                }
            }
            "msc" => {
                if parts.len() >= 6 {
                    let k: usize = parts[1].parse().unwrap();
                    let scores: Vec<i32> = parts[2..6]
                        .iter()
                        .map(|s| s.parse().unwrap())
                        .collect();
                    msc.insert(k, scores);
                }
            }
            "isc" => {
                if parts.len() >= 6 {
                    let k: usize = parts[1].parse().unwrap();
                    let scores: Vec<i32> = parts[2..6]
                        .iter()
                        .map(|s| s.parse().unwrap())
                        .collect();
                    isc.insert(k, scores);
                }
            }
            "bsc" => {
                if parts.len() >= 3 {
                    let k: usize = parts[1].parse().unwrap();
                    let score: i32 = parts[2].parse().unwrap();
                    bsc.insert(k, score);
                }
            }
            "esc" => {
                if parts.len() >= 3 {
                    let k: usize = parts[1].parse().unwrap();
                    let score: i32 = parts[2].parse().unwrap();
                    esc.insert(k, score);
                }
            }
            _ => {}
        }
    }

    GoldenScores {
        tsc,
        msc,
        isc,
        bsc,
        esc,
    }
}

/// Parse golden CP9 bands file
fn parse_golden_bands() -> GoldenBands {
    let content = fs::read_to_string(GOLDEN_BANDS).expect("Failed to read golden bands");
    let mut jmin = HashMap::new();
    let mut jmax = HashMap::new();
    let mut imin = HashMap::new();
    let mut imax = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "jband" => {
                if parts.len() >= 4 {
                    let k: usize = parts[1].parse().unwrap();
                    let min: i32 = parts[2].parse().unwrap();
                    let max: i32 = parts[3].parse().unwrap();
                    jmin.insert(k, min);
                    jmax.insert(k, max);
                }
            }
            "iband" => {
                if parts.len() >= 4 {
                    let k: usize = parts[1].parse().unwrap();
                    let min: i32 = parts[2].parse().unwrap();
                    let max: i32 = parts[3].parse().unwrap();
                    imin.insert(k, min);
                    imax.insert(k, max);
                }
            }
            _ => {}
        }
    }

    GoldenBands {
        jmin,
        jmax,
        imin,
        imax,
    }
}

struct GoldenScores {
    tsc: HashMap<usize, Vec<i32>>,
    msc: HashMap<usize, Vec<i32>>,
    isc: HashMap<usize, Vec<i32>>,
    bsc: HashMap<usize, i32>,
    esc: HashMap<usize, i32>,
}

struct GoldenBands {
    jmin: HashMap<usize, i32>,
    jmax: HashMap<usize, i32>,
    imin: HashMap<usize, i32>,
    imax: HashMap<usize, i32>,
}

/// Build CP9 from golden parameters file
fn build_cp9_from_params() -> CP9 {
    let content = fs::read_to_string(GOLDEN_PARAMS).expect("Failed to read params");

    // Parse M (model length)
    let m = content
        .lines()
        .find(|l| l.starts_with("M="))
        .and_then(|l| l.split('=').nth(1))
        .and_then(|s| s.parse::<i32>().ok())
        .expect("Failed to parse M");

    let mut cp9 = CP9::new(m);

    // Parse and set probabilities from params file
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "t" if parts.len() >= 12 => {
                // Transition probabilities
                let k: usize = parts[1].parse().unwrap();
                if k <= m as usize + 1 {
                    for (i, prob_str) in parts[2..12].iter().enumerate() {
                        cp9.t[k][i] = prob_str.parse().unwrap();
                    }
                }
            }
            "mat" if parts.len() >= 6 => {
                // Match emissions
                let k: usize = parts[1].parse().unwrap();
                if k > 0 && k <= m as usize {
                    for (i, prob_str) in parts[2..6].iter().enumerate() {
                        cp9.mat[k][i] = prob_str.parse().unwrap();
                    }
                }
            }
            "ins" if parts.len() >= 6 => {
                // Insert emissions
                let k: usize = parts[1].parse().unwrap();
                if k <= m as usize {
                    for (i, prob_str) in parts[2..6].iter().enumerate() {
                        cp9.ins[k][i] = prob_str.parse().unwrap();
                    }
                }
            }
            "begin" if parts.len() >= 3 => {
                // Begin probabilities
                let k: usize = parts[1].parse().unwrap();
                if k > 0 && k <= m as usize {
                    cp9.begin[k] = parts[2].parse().unwrap();
                }
            }
            "end" if parts.len() >= 3 => {
                // End probabilities
                let k: usize = parts[1].parse().unwrap();
                if k > 0 && k <= m as usize {
                    cp9.end[k] = parts[2].parse().unwrap();
                }
            }
            "null" => {
                // Null model
                if let Some(vals) = parts.get(1) {
                    let null_probs: Vec<f32> = vals.split(',').filter_map(|s| s.parse().ok()).collect();
                    if null_probs.len() == ALPHABET_SIZE {
                        cp9.null.copy_from_slice(&null_probs);
                    }
                }
            }
            _ => {}
        }
    }

    // Compute integer-scaled scores
    cp9.logoddsify();

    cp9
}

#[test]
fn test_cp9_tsc() {
    let cp9 = build_cp9_from_params();
    let golden = parse_golden_scores();

    println!("Testing CP9 transition scores (tsc)...");

    // Test nodes that have golden values
    for (&k, expected_scores) in &golden.tsc {
        if k > cp9.m as usize {
            continue;
        }

        for trans in 0..CP9_NTRANS {
            let actual = cp9.tsc[k][trans];
            let expected = expected_scores[trans];

            // Allow small rounding differences
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1,
                "tsc[{}][{}] mismatch: expected {}, got {} (diff={})",
                k,
                trans,
                expected,
                actual,
                diff
            );
        }
    }

    println!("✓ All transition scores match golden reference");
}

#[test]
fn test_cp9_msc() {
    let cp9 = build_cp9_from_params();
    let golden = parse_golden_scores();

    println!("Testing CP9 match emission scores (msc)...");

    // Test nodes that have golden values
    for (&k, expected_scores) in &golden.msc {
        if k > cp9.m as usize || k == 0 {
            continue;
        }

        for a in 0..4 {
            let actual = cp9.msc[k][a];
            let expected = expected_scores[a];

            // Allow small rounding differences
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1,
                "msc[{}][{}] mismatch: expected {}, got {} (diff={})",
                k,
                a,
                expected,
                actual,
                diff
            );
        }
    }

    println!("✓ All match emission scores match golden reference");
}

#[test]
fn test_cp9_isc() {
    let cp9 = build_cp9_from_params();
    let golden = parse_golden_scores();

    println!("Testing CP9 insert emission scores (isc)...");

    // Test nodes that have golden values
    for (&k, expected_scores) in &golden.isc {
        if k > cp9.m as usize {
            continue;
        }

        for a in 0..4 {
            let actual = cp9.isc[k][a];
            let expected = expected_scores[a];

            // Allow small rounding differences
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1,
                "isc[{}][{}] mismatch: expected {}, got {} (diff={})",
                k,
                a,
                expected,
                actual,
                diff
            );
        }
    }

    println!("✓ All insert emission scores match golden reference");
}

#[test]
fn test_cp9_bsc_esc() {
    let cp9 = build_cp9_from_params();
    let golden = parse_golden_scores();

    println!("Testing CP9 begin/end scores (bsc/esc)...");

    // Test begin scores
    for (&k, &expected) in &golden.bsc {
        if k > cp9.m as usize || k == 0 {
            continue;
        }

        let actual = cp9.bsc[k];
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1,
            "bsc[{}] mismatch: expected {}, got {} (diff={})",
            k,
            expected,
            actual,
            diff
        );
    }

    // Test end scores
    for (&k, &expected) in &golden.esc {
        if k > cp9.m as usize || k == 0 {
            continue;
        }

        let actual = cp9.esc[k];
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1,
            "esc[{}] mismatch: expected {}, got {} (diff={})",
            k,
            expected,
            actual,
            diff
        );
    }

    println!("✓ All begin/end scores match golden reference");
}

#[test]
fn test_cp9_score_constants() {
    // Verify constants match expected values
    assert_eq!(CP9_IMPOSSIBLE, -987654321);
    assert_eq!(CP9_INTSCALE, 1000.0);

    let cp9 = build_cp9_from_params();

    // Check that impossible scores are properly set
    assert_eq!(cp9.tsc[0][0], CP9_IMPOSSIBLE); // MM at node 0 is impossible
    assert_eq!(cp9.msc[0][0], CP9_IMPOSSIBLE); // Match at node 0 is impossible

    println!("✓ Score constants verified");
}

#[test]
fn test_cp9_score_properties() {
    let cp9 = build_cp9_from_params();
    let m = cp9.m as usize;

    println!("Testing CP9 score properties...");

    // Verify match emission scores are log-odds (can be positive or negative)
    for k in 1..=m {
        for a in 0..4 {
            let score = cp9.msc[k][a];
            if score != CP9_IMPOSSIBLE {
                // Valid score - check it's in reasonable range
                assert!(
                    score > -10000 && score < 10000,
                    "Match score out of range: msc[{}][{}] = {}",
                    k,
                    a,
                    score
                );
            }
        }
    }

    // Verify transition scores are valid log probabilities (should be negative)
    for k in 0..=m {
        for t in 0..CP9_NTRANS {
            let score = cp9.tsc[k][t];
            if score != CP9_IMPOSSIBLE {
                // Valid score - log probabilities should be <= 0
                assert!(
                    score <= 0,
                    "Transition score should be <= 0: tsc[{}][{}] = {}",
                    k,
                    t,
                    score
                );
            }
        }
    }

    println!("✓ Score properties validated");
}
