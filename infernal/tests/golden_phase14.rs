//! Phase 14: cmsearch E2E Tests
//!
//! End-to-end tests for the cmsearch pipeline using golden reference files.

use infernal::cm_file_read;
use infernal::cmsearch::{cmsearch, format_tblout};
use std::path::PathBuf;

/// Get path to test CM file (tRNA model)
fn get_cm_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("trna-5.cm")
}

/// Test tRNA sequence (76 nt)
const TRNA_SEQ: &str = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA";

#[test]
fn test_cmsearch_trna_score() {
    // Test that cmsearch finds tRNA with positive score
    let cm_path = get_cm_path();
    let cm = cm_file_read(&cm_path.to_str().unwrap())
        .expect("Failed to read CM file");

    let result = cmsearch(&cm, TRNA_SEQ, "tRNA-1")
        .expect("cmsearch failed");

    assert_eq!(result.hits.len(), 1, "Should find exactly 1 hit");

    let hit = &result.hits[0];
    // Score should be positive and reasonable for a tRNA hit
    assert!(hit.score > 40.0, "Score should be > 40 bits, got {}", hit.score);
    assert!(hit.score < 100.0, "Score should be < 100 bits, got {}", hit.score);
}

#[test]
fn test_cmsearch_trna_tblout() {
    // Test that tblout formatting is correct
    let cm_path = get_cm_path();
    let cm = cm_file_read(&cm_path.to_str().unwrap())
        .expect("Failed to read CM file");

    let result = cmsearch(&cm, TRNA_SEQ, "tRNA-1")
        .expect("cmsearch failed");

    let tblout = format_tblout(&[result]);

    // Check header lines
    assert!(tblout.contains("#target name"), "Should have header, got: {}", tblout);
    assert!(tblout.contains("#---"), "Should have separator line");

    // Check data line exists
    let lines: Vec<&str> = tblout.lines()
        .filter(|l| !l.starts_with('#'))
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), 1, "Should have exactly 1 data line");

    // Check data line contains expected fields
    let data_line = lines[0];
    assert!(data_line.contains("tRNA-1"), "Should contain target name");
    assert!(data_line.contains("trna.5-1"), "Should contain query name");
    assert!(data_line.contains("+"), "Should contain strand");
}

#[test]
fn test_cmsearch_hit_count() {
    // Test that we find at least one hit for valid tRNA sequence
    let cm_path = get_cm_path();
    let cm = cm_file_read(&cm_path.to_str().unwrap())
        .expect("Failed to read CM file");

    let result = cmsearch(&cm, TRNA_SEQ, "tRNA-1")
        .expect("cmsearch failed");

    assert!(!result.hits.is_empty(), "Should find at least one hit");
    assert_eq!(result.model_name, "trna.5-1", "Model name should be 'trna.5-1'");
    assert_eq!(result.target_name, "tRNA-1", "Target name should be 'tRNA-1'");
    assert_eq!(result.target_len, TRNA_SEQ.len() as i32,
               "Target length should match sequence length");
}

#[test]
fn test_cmsearch_strand() {
    // Test that strand detection works (for now, always +)
    let cm_path = get_cm_path();
    let cm = cm_file_read(&cm_path.to_str().unwrap())
        .expect("Failed to read CM file");

    let result = cmsearch(&cm, TRNA_SEQ, "tRNA-1")
        .expect("cmsearch failed");

    let tblout = format_tblout(&[result]);
    let data_line: Vec<&str> = tblout.lines()
        .filter(|l| !l.starts_with('#'))
        .filter(|l| !l.trim().is_empty())
        .collect();

    assert_eq!(data_line.len(), 1);
    // Strand should be '+' (column 10 in tblout format, 0-indexed field 9)
    let fields: Vec<&str> = data_line[0].split_whitespace().collect();
    assert!(fields.len() >= 10, "Should have at least 10 fields, got {}", fields.len());
    assert_eq!(fields[9], "+", "Strand should be '+'");
}

#[test]
fn test_cmsearch_alignment_bounds() {
    // Test that alignment bounds are reasonable
    let cm_path = get_cm_path();
    let cm = cm_file_read(&cm_path.to_str().unwrap())
        .expect("Failed to read CM file");
    
    let result = cmsearch(&cm, TRNA_SEQ, "tRNA-1")
        .expect("cmsearch failed");
    
    let hit = &result.hits[0];
    assert!(hit.start >= 1, "Start should be >= 1");
    assert!(hit.end <= TRNA_SEQ.len() as i32, "End should be <= sequence length");
    assert!(hit.start <= hit.end, "Start should be <= end");
}
