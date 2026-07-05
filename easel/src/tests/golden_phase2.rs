//! Phase 2 Golden Tests - Alphabet
//!
//! Tests for RNA alphabet implementation including digitization
//! and symbol mapping.
//! Golden data generated from Easel alphabet module.

use std::fs;
use crate::alphabet::EslAlphabet;

fn load_golden(filename: &str) -> String {
    fs::read_to_string(format!("../tests/golden/phase2/{}", filename))
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

fn parse_i32(content: &str, key: &str) -> Option<i32> {
    parse_key_value(content, key)?.parse().ok()
}

fn parse_i32_array(content: &str, key: &str) -> Vec<i32> {
    parse_key_value(content, key)
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_u8_array(content: &str, key: &str) -> Vec<u8> {
    parse_key_value(content, key)
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

// =====================================================================
// Alphabet Dimension Tests
// =====================================================================

#[test]
fn test_rna_alphabet_type() {
    let golden = load_golden("rna_alphabet.txt");
    let typ = parse_i32(&golden, "type").expect("type not found");

    let abc = EslAlphabet::rna();
    assert_eq!(abc.type_, typ, "RNA alphabet type should match golden");
}

#[test]
fn test_rna_alphabet_k() {
    let golden = load_golden("rna_alphabet.txt");
    let k = parse_i32(&golden, "K").expect("K not found");

    let abc = EslAlphabet::rna();
    assert_eq!(abc.K, k, "K should match golden");
}

#[test]
fn test_rna_alphabet_kp() {
    let golden = load_golden("rna_alphabet.txt");
    let kp = parse_i32(&golden, "Kp").expect("Kp not found");

    let abc = EslAlphabet::rna();
    assert_eq!(abc.Kp, kp, "Kp should match golden");
}

// =====================================================================
// Symbol Tests
// =====================================================================

#[test]
fn test_rna_alphabet_sym() {
    let golden = load_golden("rna_alphabet.txt");
    let sym_golden = parse_key_value(&golden, "sym").expect("sym not found");

    let abc = EslAlphabet::rna();
    let sym: String = abc.sym.iter().collect();

    assert_eq!(sym, sym_golden, "sym should match golden");
    assert_eq!(sym.len(), abc.Kp as usize, "sym length should match Kp");
}

// =====================================================================
// Input Map (Digitization) Tests
// =====================================================================

#[test]
fn test_rna_alphabet_inmap() {
    let golden = load_golden("rna_alphabet.txt");
    let inmap_golden = parse_u8_array(&golden, "inmap");

    assert_eq!(inmap_golden.len(), 128, "golden inmap should have 128 entries");

    let abc = EslAlphabet::rna();

    // Compare all 128 entries
    for i in 0..128 {
        assert_eq!(
            abc.inmap[i], inmap_golden[i],
            "inmap[{}] mismatch: got {}, expected {} (char '{}')",
            i, abc.inmap[i], inmap_golden[i],
            if i >= 32 && i < 127 { i as u8 as char } else { '?' }
        );
    }
}

// =====================================================================
// Complement Tests
// =====================================================================

#[test]
fn test_rna_alphabet_complement() {
    let golden = load_golden("rna_alphabet.txt");
    let complement_golden = parse_u8_array(&golden, "complement");

    assert_eq!(complement_golden.len(), 18, "golden complement should have 18 entries");

    let abc = EslAlphabet::rna();

    // Compare all 18 entries
    for i in 0..18 {
        assert_eq!(
            abc.complement[i], complement_golden[i],
            "complement[{}] mismatch: got {}, expected {} (symbol '{}')",
            i, abc.complement[i], complement_golden[i], abc.sym[i]
        );
    }
}

// =====================================================================
// Digitize Tests
// =====================================================================

#[test]
fn test_digitize_canonical() {
    let abc = EslAlphabet::rna();
    let dsq = abc.digitize("ACGU");
    assert_eq!(dsq, vec![0, 1, 2, 3], "ACGU should digitize to [0,1,2,3]");
}

#[test]
fn test_digitize_ambiguous() {
    let abc = EslAlphabet::rna();
    let dsq = abc.digitize("NNNN");
    assert_eq!(dsq, vec![15, 15, 15, 15], "NNNN should digitize to [15,15,15,15]");
}

#[test]
fn test_digitize_mixed() {
    let abc = EslAlphabet::rna();
    let dsq = abc.digitize("ACGN");
    assert_eq!(dsq, vec![0, 1, 2, 15], "ACGN should digitize to [0,1,2,15]");
}

#[test]
fn test_digitize_lowercase() {
    let abc = EslAlphabet::rna();
    let dsq = abc.digitize("acgu");
    assert_eq!(dsq, vec![0, 1, 2, 3], "acgu should digitize to [0,1,2,3]");
}

#[test]
fn test_digitize_degeneracy() {
    let abc = EslAlphabet::rna();
    let dsq = abc.digitize("RYSWKM");
    assert_eq!(dsq, vec![5, 6, 9, 10, 8, 7], "RYSWKM should digitize to [5,6,9,10,8,7]");
}

#[test]
fn test_digitize_gap() {
    let abc = EslAlphabet::rna();
    let dsq = abc.digitize("A-C");
    assert_eq!(dsq, vec![0, 4, 1], "A-C should digitize to [0,4,1]");
}

// =====================================================================
// Textize Tests
// =====================================================================

#[test]
fn test_textize_canonical() {
    let abc = EslAlphabet::rna();
    let text = abc.textize(&[0, 1, 2, 3]);
    assert_eq!(text, "ACGU", "textize should produce ACGU");
}

#[test]
fn test_textize_degeneracy() {
    let abc = EslAlphabet::rna();
    let text = abc.textize(&[5, 6, 7, 8, 9, 10, 15]);
    assert_eq!(text, "RYMKSWN", "textize should produce RYMKSWN");
}

// =====================================================================
// Degeneracy Code Tests
// =====================================================================

#[test]
fn test_degeneracy_codes() {
    let abc = EslAlphabet::rna();

    // R = A or G (purine) - index 5
    assert_eq!(abc.sym[5], 'R');
    // Y = C or U (pyrimidine) - index 6
    assert_eq!(abc.sym[6], 'Y');
    // M = A or C - index 7
    assert_eq!(abc.sym[7], 'M');
    // K = G or U - index 8
    assert_eq!(abc.sym[8], 'K');
    // S = G or C - index 9
    assert_eq!(abc.sym[9], 'S');
    // W = A or U - index 10
    assert_eq!(abc.sym[10], 'W');
}

// =====================================================================
// Round-trip Tests
// =====================================================================

#[test]
fn test_roundtrip() {
    let abc = EslAlphabet::rna();
    let original = "ACGURYSWKN";
    let dsq = abc.digitize(original);
    let recovered = abc.textize(&dsq);
    assert_eq!(recovered, original, "round-trip should preserve sequence");
}
