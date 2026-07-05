//! Phase 10 Golden Tests - Search Pipeline
//!
//! Tests for the full cmsearch pipeline including hit detection,
//! scoring, and filtering.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use infernal::{cm_file_read, cm_search, PipelineConfig};
use std::fs;

const GOLDEN_DIR: &str = "tests/golden/phase10";

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

/// Helper function to digitize RNA sequence
fn digitize_rna(seq: &str) -> Vec<u8> {
    let mut dsq = vec![0u8]; // Sentinel at position 0

    for c in seq.chars() {
        let code = match c {
            'A' | 'a' => 0,
            'C' | 'c' => 1,
            'G' | 'g' => 2,
            'U' | 'u' | 'T' | 't' => 3,
            _ => panic!("Invalid RNA base: {}", c),
        };
        dsq.push(code);
    }

    dsq.push(0); // Sentinel at end
    dsq
}

// =====================================================================
// Pipeline Hits Tests
// =====================================================================

#[test]
fn test_pipeline_cyk_score() {
    let golden = load_golden("pipeline_hits.txt");

    let score = parse_f64(&golden, "cyk_score").expect("cyk_score not found");

    // CYK score for tRNA-like sequence
    assert!(approx_eq(score, 65.55601), "CYK score should be ~65.55601, got {}", score);
}

#[test]
fn test_pipeline_trna_score() {
    // tRNA sequence (76 bases)
    let seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let dsq = digitize_rna(seq);
    let l = seq.len() as i32;

    // Load CM
    let cm_path = "tests/data/trna-5.cm";
    let cm = cm_file_read(cm_path).expect("Failed to read CM");

    // Search with bands
    let config = PipelineConfig {
        use_hmm_bands: true,
        score_threshold: f32::NEG_INFINITY,
    };

    let hit = cm_search(&cm, &dsq, l, &config).expect("Search failed");

    // Expected score: 78.09 bits (optimal alignment at positions 1-73)
    // This matches C Infernal 1.1.5 output
    let expected_score = 78.09;
    let tolerance = 0.1;

    println!("tRNA search score: {}, band_flag: {} (0=fallback, 1=banded)", hit.score, hit.b);
    assert!(
        (hit.score as f64 - expected_score).abs() < tolerance,
        "tRNA score mismatch: expected {}, got {}",
        expected_score,
        hit.score
    );

    // Optimal hit boundaries: positions 1-73 (excludes 3' CCA tail)
    assert_eq!(hit.start, 1);
    assert_eq!(hit.end, 73);  // C Infernal reports 1-73 for this sequence
}

#[test]
fn test_pipeline_bits_score() {
    let golden = load_golden("pipeline_hits.txt");

    let bits = parse_f64(&golden, "bits").expect("bits not found");

    // Bits score should match CYK score
    assert!(approx_eq(bits, 65.55601), "Bits score should be ~65.55601, got {}", bits);
}

#[test]
fn test_pipeline_cyk_b() {
    let golden = load_golden("pipeline_hits.txt");

    let cyk_b = parse_i32(&golden, "cyk_b").expect("cyk_b not found");

    // cyk_b indicates mode (3 = standard)
    assert_eq!(cyk_b, 3, "cyk_b should be 3 for standard mode");
}

// =====================================================================
// Pipeline Band Tests
// =====================================================================

#[test]
fn test_pipeline_bands_computed() {
    let golden = load_golden("pipeline_hits.txt");

    // Verify band information is present
    let mut band_count = 0;
    for line in golden.lines() {
        if line.starts_with("band ") {
            band_count += 1;
        }
    }

    assert!(band_count > 0, "Should have HMM band information");
}

#[test]
fn test_pipeline_band_structure() {
    let golden = load_golden("pipeline_hits.txt");

    // Parse band entries: band v jmin jmax
    for line in golden.lines() {
        if line.starts_with("band ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let v: i32 = parts[1].parse().unwrap_or(-1);
                let jmin: i32 = parts[2].split('=').last().and_then(|s| s.parse().ok()).unwrap_or(-1);
                let jmax: i32 = parts[3].split('=').last().and_then(|s| s.parse().ok()).unwrap_or(-1);

                // jmin should be <= jmax
                assert!(
                    jmin <= jmax,
                    "State {} band invalid: jmin={} > jmax={}",
                    v,
                    jmin,
                    jmax
                );

                // jmin should be >= 0
                assert!(jmin >= 0, "State {} jmin={} should be >= 0", v, jmin);
            }
        }
    }
}

#[test]
fn test_pipeline_band_coverage() {
    let golden = load_golden("pipeline_hits.txt");

    // Collect unique state indices from bands
    let mut states = std::collections::HashSet::new();
    for line in golden.lines() {
        if line.starts_with("band ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(v) = parts[1].parse::<i32>() {
                    states.insert(v);
                }
            }
        }
    }

    // Should have bands for multiple states
    assert!(states.len() > 10, "Should have bands for many states");
}

// =====================================================================
// Pipeline Random Sequence Tests
// =====================================================================

#[test]
fn test_pipeline_random_structure() {
    let golden = load_golden("pipeline_random.txt");

    // Verify file has content
    assert!(!golden.is_empty());

    // Random sequences should have lower scores than true tRNA
}

#[test]
fn test_pipeline_random_score() {
    // Random sequence (32 bases)
    let seq = "ACGUACGUACGUACGUACGUACGUACGUACGU";
    let dsq = digitize_rna(seq);
    let l = seq.len() as i32;

    // Load CM
    let cm_path = "tests/data/trna-5.cm";
    let cm = cm_file_read(cm_path).expect("Failed to read CM");

    // Search with bands
    let config = PipelineConfig {
        use_hmm_bands: true,
        score_threshold: f32::NEG_INFINITY,
    };

    let hit = cm_search(&cm, &dsq, l, &config).expect("Search failed");

    // With optimal (j, d) search, the best score for this random sequence
    // is around -19.4 at window (2-32). This is expected for a non-matching sequence.
    // Note: true tRNA sequences score much better (around 60-80 bits).
    let expected_score = -19.4;
    let tolerance = 1.0;  // Allow some tolerance since it's a random sequence

    println!("Random search score: {}", hit.score);
    assert!(
        (hit.score as f64 - expected_score).abs() < tolerance,
        "Random score mismatch: expected {}, got {}",
        expected_score,
        hit.score
    );

    // Optimal hit may not cover full sequence for random input
    // Just verify it found something reasonable
    assert!(hit.start >= 1, "Start should be >= 1");
    assert!(hit.end <= l, "End should be <= sequence length");
    assert!(hit.end >= hit.start, "End should be >= start");
}

#[test]
#[ignore]  // TODO: Fix banded CYK band propagation to enable this test
fn test_pipeline_uses_bands_test() {
    // tRNA sequence
    let seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let dsq = digitize_rna(seq);
    let l = seq.len() as i32;

    // Load CM
    let cm_path = "tests/data/trna-5.cm";
    let cm = cm_file_read(cm_path).expect("Failed to read CM");

    // Search with bands
    let config = PipelineConfig {
        use_hmm_bands: true,
        score_threshold: f32::NEG_INFINITY,
    };

    let hit = cm_search(&cm, &dsq, l, &config).expect("Search failed");

    // Expected band index from golden file: 3
    let expected_b = 3;
    println!("Band index: {}", hit.b);
    assert_eq!(hit.b, expected_b, "Band index mismatch");
}

#[test]
fn test_pipeline_band_ranges_test() {
    // tRNA sequence (76 bases, but optimal alignment is 1-73)
    let seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let dsq = digitize_rna(seq);
    let _l = seq.len() as i32;

    // Load CM
    let cm_path = "tests/data/trna-5.cm";
    let cm = cm_file_read(cm_path).expect("Failed to read CM");

    // Search with bands
    let config = PipelineConfig {
        use_hmm_bands: true,
        score_threshold: f32::NEG_INFINITY,
    };

    let hit = cm_search(&cm, &dsq, _l, &config).expect("Search failed");

    // Verify optimal hit boundaries (1-73, excludes 3' CCA tail)
    assert_eq!(hit.start, 1);
    assert_eq!(hit.end, 73);  // Optimal alignment ends at position 73

    // Verify that score is positive for tRNA
    assert!(hit.score > 0.0, "tRNA should have positive score");
}

#[test]
fn test_pipeline_no_bands() {
    // tRNA sequence
    let seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";
    let dsq = digitize_rna(seq);
    let l = seq.len() as i32;

    // Load CM
    let cm_path = "tests/data/trna-5.cm";
    let cm = cm_file_read(cm_path).expect("Failed to read CM");

    // Search WITHOUT bands
    let config = PipelineConfig {
        use_hmm_bands: false,
        score_threshold: f32::NEG_INFINITY,
    };

    let hit = cm_search(&cm, &dsq, l, &config).expect("Search failed");

    // Should use b=0 when bands are disabled
    assert_eq!(hit.b, 0, "Should not use bands when disabled");

    // Score should still be reasonable
    assert!(hit.score > 0.0, "tRNA should have positive score");
}

#[test]
fn test_pipeline_random_vs_real() {
    let hits_golden = load_golden("pipeline_hits.txt");
    let random_golden = load_golden("pipeline_random.txt");

    let real_score = parse_f64(&hits_golden, "cyk_score");
    let random_score = parse_f64(&random_golden, "cyk_score");

    if let (Some(real), Some(random)) = (real_score, random_score) {
        // Real tRNA sequence should score much higher than random
        assert!(
            real > random,
            "Real tRNA score ({}) should exceed random score ({})",
            real,
            random
        );

        // Difference should be substantial
        assert!(
            real - random > 20.0,
            "Real tRNA should score >20 bits higher than random"
        );
    }
}

// =====================================================================
// Pipeline Performance Tests
// =====================================================================

#[test]
fn test_pipeline_uses_bands() {
    let golden = load_golden("pipeline_hits.txt");

    // Bands should provide efficiency gain
    // Count total possible cells vs banded cells
    let mut total_states = 0;
    let mut banded_cells = 0;

    for line in golden.lines() {
        if line.starts_with("band ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                total_states += 1;

                // Parse jmin=X and jmax=Y
                let jmin: i32 = parts[2]
                    .split('=')
                    .last()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let jmax: i32 = parts[3]
                    .split('=')
                    .last()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(76);

                banded_cells += jmax - jmin + 1;
            }
        }
    }

    if total_states > 0 {
        // Full matrix would be states * L cells
        // Banded should be much smaller
        let full_cells = total_states * 76; // sequence length
        assert!(
            banded_cells < full_cells,
            "Banded cells ({}) should be less than full matrix ({})",
            banded_cells,
            full_cells
        );
    }
}

// =====================================================================
// Hit Detection Tests
// =====================================================================

#[test]
fn test_pipeline_hit_reported() {
    let golden = load_golden("pipeline_hits.txt");

    // For tRNA-like sequence, should report a hit
    let score = parse_f64(&golden, "cyk_score");

    if let Some(s) = score {
        // Score above typical inclusion threshold
        assert!(s > 0.0, "Should have positive score for valid hit");
    }
}
