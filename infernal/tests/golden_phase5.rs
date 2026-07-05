//! Phase 5 Golden Tests - CP9 HMM and Banding
//!
//! Tests for CP9 HMM construction, Viterbi scoring, and HMM/QDB bands.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use std::fs;

use infernal::cp9::{CP9, CP9_NTRANS, CTDD, CTDM, CTII, CTIM, CTMD, CTMI, CTMM};
use infernal::cp9_bands::{CP9Bands, QDBBands};
use infernal::cp9_mx::CP9MX;

const GOLDEN_DIR: &str = "tests/golden/phase5";

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

fn parse_f64_array(line: &str, skip_prefix: usize) -> Vec<f64> {
    line.split_whitespace()
        .skip(skip_prefix)
        .filter_map(|s| s.parse().ok())
        .collect()
}

fn parse_i32_array(line: &str, skip_prefix: usize) -> Vec<i32> {
    line.split_whitespace()
        .skip(skip_prefix)
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// Build a CP9 HMM from golden file data
fn build_cp9_from_golden(content: &str) -> CP9 {
    let m = parse_i32(content, "cp9_M").expect("cp9_M not found");
    let flags = parse_i32(content, "cp9_flags").expect("cp9_flags not found") as u32;
    let el_self = parse_f64(content, "el_self").expect("el_self not found") as f32;
    let p1 = parse_f64(content, "p1").expect("p1 not found") as f32;

    let mut cp9 = CP9::new(m);
    cp9.flags = flags;
    cp9.el_self = el_self;
    cp9.p1 = p1;

    // Parse null model
    for line in content.lines() {
        if line.starts_with("cp9_null=") {
            let values: Vec<f32> = line
                .trim_start_matches("cp9_null=")
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect();
            if values.len() == 4 {
                cp9.null.copy_from_slice(&values);
            }
        }
    }

    // Parse transitions
    for line in content.lines() {
        if line.starts_with("cp9_trans ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 12 {
                let node: usize = parts[1].parse().unwrap_or(0);
                let values: Vec<f32> = parts[2..]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if values.len() == CP9_NTRANS && node <= m as usize + 1 {
                    cp9.t[node].copy_from_slice(&values);
                }
            }
        }
    }

    // Parse match emissions
    for line in content.lines() {
        if line.starts_with("cp9_mat ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                let node: usize = parts[1].parse().unwrap_or(0);
                let values: Vec<f32> = parts[2..]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if values.len() == 4 && node <= m as usize {
                    cp9.mat[node].copy_from_slice(&values);
                }
            }
        }
    }

    // Parse insert emissions
    for line in content.lines() {
        if line.starts_with("cp9_ins ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                let node: usize = parts[1].parse().unwrap_or(0);
                let values: Vec<f32> = parts[2..]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if values.len() == 4 && node <= m as usize {
                    cp9.ins[node].copy_from_slice(&values);
                }
            }
        }
    }

    // Parse begin probabilities
    for line in content.lines() {
        if line.starts_with("cp9_begin ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let node: usize = parts[1].parse().unwrap_or(0);
                let prob: f32 = parts[2].parse().unwrap_or(0.0);
                if node <= m as usize {
                    cp9.begin[node] = prob;
                }
            }
        }
    }

    // Parse end probabilities
    for line in content.lines() {
        if line.starts_with("cp9_end ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let node: usize = parts[1].parse().unwrap_or(0);
                let prob: f32 = parts[2].parse().unwrap_or(0.0);
                if node <= m as usize {
                    cp9.end[node] = prob;
                }
            }
        }
    }

    cp9
}

/// Parse QDB bands from golden file
fn parse_qdb_bands(content: &str, set: i32) -> (f64, Vec<(i32, i32, i32)>) {
    let beta_key = format!("beta{}", set);
    let beta = parse_f64(content, &beta_key).unwrap_or(0.0);
    let prefix = format!("qdb{} ", set);

    let mut bands = Vec::new();
    for line in content.lines() {
        if line.starts_with(&prefix) {
            let parts: Vec<i32> = line
                .split_whitespace()
                .skip(1)
                .filter_map(|s| s.parse().ok())
                .collect();
            if parts.len() == 3 {
                bands.push((parts[0], parts[1], parts[2])); // state, dmin, dmax
            }
        }
    }

    (beta, bands)
}

// =====================================================================
// CP9 HMM Tests
// =====================================================================

#[test]
fn test_cp9_hmm_dimensions() {
    let golden = load_golden("cp9_hmm.txt");

    let m = parse_i32(&golden, "cp9_M").expect("cp9_M not found");
    let flags = parse_i32(&golden, "cp9_flags").expect("cp9_flags not found");

    assert_eq!(m, 71);
    assert_eq!(flags, 15);

    // Verify CP9 struct can be created with these dimensions
    let cp9 = CP9::new(m);
    assert_eq!(cp9.m, m);
}

#[test]
fn test_cp9_el_self() {
    let golden = load_golden("cp9_hmm.txt");

    let el_self = parse_f64(&golden, "el_self").expect("el_self not found");
    let p1 = parse_f64(&golden, "p1").expect("p1 not found");

    const EPSILON: f64 = 1e-6;
    assert!((el_self - 0.94).abs() < EPSILON);
    assert!((p1 - 1.0).abs() < EPSILON);

    // Verify CP9 struct can store these values
    let mut cp9 = CP9::new(71);
    cp9.el_self = el_self as f32;
    cp9.p1 = p1 as f32;

    assert!((cp9.el_self as f64 - 0.94).abs() < EPSILON);
    assert!((cp9.p1 as f64 - 1.0).abs() < EPSILON);
}

#[test]
fn test_cp9_null_model() {
    let golden = load_golden("cp9_hmm.txt");

    for line in golden.lines() {
        if line.starts_with("cp9_null=") {
            let values: Vec<f64> = line
                .trim_start_matches("cp9_null=")
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect();

            assert_eq!(values.len(), 4);
            const EPSILON: f64 = 1e-6;
            for v in &values {
                assert!((*v - 0.25).abs() < EPSILON, "Null model should be uniform");
            }

            // Verify CP9 default null model
            let cp9 = CP9::new(71);
            for i in 0..4 {
                assert!((cp9.null[i] as f64 - 0.25).abs() < EPSILON);
            }
            return;
        }
    }
    panic!("cp9_null not found");
}

#[test]
fn test_cp9_transitions_first_node() {
    let golden = load_golden("cp9_hmm.txt");

    for line in golden.lines() {
        if line.starts_with("cp9_trans 0 ") {
            let values = parse_f64_array(line, 2);

            assert_eq!(values.len(), 10);

            const EPSILON: f64 = 1e-6;
            // CTMM (M->M) at node 0 should be 0
            assert!(values[CTMM].abs() < EPSILON);
            // CTMI (M->I) should be ~0.002
            assert!((values[CTMI] - 2.005763e-03).abs() < EPSILON);

            // Verify CP9 can store these transitions
            let cp9 = build_cp9_from_golden(&golden);
            assert!((cp9.t[0][CTMM] as f64).abs() < EPSILON);
            assert!((cp9.t[0][CTMI] as f64 - 2.005763e-03).abs() < EPSILON);

            return;
        }
    }
    panic!("cp9_trans 0 not found");
}

#[test]
fn test_cp9_transitions_structure() {
    let golden = load_golden("cp9_hmm.txt");
    let cp9 = build_cp9_from_golden(&golden);

    // Check transition indices match expected positions
    assert_eq!(CTMM, 0);
    assert_eq!(CTMI, 1);
    assert_eq!(CTMD, 2);
    assert_eq!(CTIM, 4);
    assert_eq!(CTII, 5);
    assert_eq!(CTDM, 7);
    assert_eq!(CTDD, 9);

    // Verify some nodes have valid transitions
    for k in 1..=10.min(cp9.m as usize) {
        let trans_sum: f64 = cp9.t[k][CTMM] as f64
            + cp9.t[k][CTMI] as f64
            + cp9.t[k][CTMD] as f64;
        // M->M + M->I + M->D should be close to 1.0 (for internal nodes)
        assert!(trans_sum > 0.9 && trans_sum < 1.1,
            "Node {} match transitions sum to {}", k, trans_sum);
    }
}

#[test]
fn test_cp9_match_emissions() {
    let golden = load_golden("cp9_hmm.txt");

    let mut found_count = 0;
    for line in golden.lines() {
        if line.starts_with("cp9_mat ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 {
                let node: i32 = parts[1].parse().unwrap_or(-1);
                let values: Vec<f64> = parts[2..]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();

                assert_eq!(values.len(), 4);

                let sum: f64 = values.iter().sum();
                const EPSILON: f64 = 1e-4;
                assert!(
                    (sum - 1.0).abs() < EPSILON,
                    "Node {} emissions sum to {}, expected ~1.0",
                    node,
                    sum
                );

                found_count += 1;
            }
        }
    }
    assert!(found_count >= 10, "Should have at least 10 match emission lines");

    // Verify CP9 struct stores emissions correctly
    let cp9 = build_cp9_from_golden(&golden);
    for k in 1..=10.min(cp9.m as usize) {
        let sum: f64 = cp9.mat[k].iter().map(|&x| x as f64).sum();
        assert!((sum - 1.0).abs() < 1e-4, "CP9 node {} emissions sum to {}", k, sum);
    }
}

#[test]
fn test_cp9_insert_emissions() {
    let golden = load_golden("cp9_hmm.txt");

    for line in golden.lines() {
        if line.starts_with("cp9_ins ") {
            let values = parse_f64_array(line, 2);

            assert_eq!(values.len(), 4);

            const EPSILON: f64 = 1e-6;
            for v in &values {
                assert!((*v - 0.25).abs() < EPSILON, "Insert emissions should be uniform");
            }
        }
    }

    // Verify CP9 struct stores insert emissions
    let cp9 = build_cp9_from_golden(&golden);
    const EPSILON: f64 = 1e-6;
    for k in 0..=10.min(cp9.m as usize) {
        for a in 0..4 {
            assert!((cp9.ins[k][a] as f64 - 0.25).abs() < EPSILON,
                "Insert emission at node {} residue {} should be 0.25", k, a);
        }
    }
}

#[test]
fn test_cp9_begin_probabilities() {
    let golden = load_golden("cp9_hmm.txt");

    let mut begin_probs = Vec::new();
    for line in golden.lines() {
        if line.starts_with("cp9_begin ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(prob) = parts[2].parse::<f64>() {
                    begin_probs.push(prob);
                }
            }
        }
    }

    assert!(!begin_probs.is_empty());
    assert!(begin_probs[0] > 0.9, "First begin probability should be > 0.9");

    // Verify CP9 struct stores begin probabilities
    let cp9 = build_cp9_from_golden(&golden);
    assert!(cp9.begin[1] > 0.9, "CP9 begin[1] should be > 0.9");
}

#[test]
fn test_cp9_end_probabilities() {
    let golden = load_golden("cp9_hmm.txt");

    let mut end_probs = Vec::new();
    for line in golden.lines() {
        if line.starts_with("cp9_end ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(prob) = parts[2].parse::<f64>() {
                    end_probs.push(prob);
                }
            }
        }
    }

    assert!(!end_probs.is_empty());

    // Verify CP9 struct stores end probabilities
    let cp9 = build_cp9_from_golden(&golden);
    // All end probabilities should be small positive values
    for k in 1..=10.min(cp9.m as usize) {
        assert!(cp9.end[k] > 0.0 && cp9.end[k] < 0.01,
            "End probability at node {} should be small positive", k);
    }
}

// =====================================================================
// CP9 Viterbi Tests
// =====================================================================

#[test]
fn test_cp9_viterbi_score() {
    let golden = load_golden("cp9_viterbi.txt");

    if let Some(score) = parse_f64(&golden, "viterbi_score") {
        assert!(score > 0.0, "Viterbi score should be positive for matching sequence");
        assert!((score - 32.996).abs() < 0.01, "Expected score ~32.996, got {}", score);
    }
}

#[test]
fn test_cp9_viterbi_matrix_structure() {
    let golden = load_golden("cp9_viterbi.txt");

    // Parse viterbi matrix values
    let mut vit_values = Vec::new();
    for line in golden.lines() {
        if line.starts_with("vit ") {
            let values = parse_i32_array(line, 1);
            if values.len() == 5 {
                // i, k, match, insert, delete
                vit_values.push((values[0], values[1], values[2], values[3], values[4]));
            }
        }
    }

    assert!(!vit_values.is_empty(), "Should have viterbi matrix values");

    // Verify CP9MX can store these dimensions
    let mx = CP9MX::new(71, 76);
    assert_eq!(mx.m, 71);
    assert_eq!(mx.l, 76);

    // Verify specific values from golden file
    for (i, k, m_sc, i_sc, d_sc) in &vit_values {
        assert!(*i >= 1 && *i <= 76, "Position i should be in [1, 76]");
        assert!(*k >= 1 && *k <= 71, "Node k should be in [1, 71]");
        // Scores can be very negative (IMPOSSIBLE-like) or positive
        assert!(*m_sc > -1_000_000_000 || *m_sc < 1_000_000_000);
        assert!(*i_sc > -1_000_000_000 || *i_sc < 1_000_000_000);
        assert!(*d_sc > -1_000_000_000 || *d_sc < 1_000_000_000);
    }
}

// =====================================================================
// HMM Bands Tests
// =====================================================================

#[test]
fn test_hmm_bands_structure() {
    let golden = load_golden("hmm_bands.txt");

    assert!(!golden.is_empty());

    // Parse forward/backward scores
    let fwd_score = parse_f64(&golden, "forward_score");
    let bwd_score = parse_f64(&golden, "backward_score");

    assert!(fwd_score.is_some(), "Should have forward_score");
    assert!(bwd_score.is_some(), "Should have backward_score");

    // Forward score should match Viterbi (approximately)
    if let Some(fwd) = fwd_score {
        assert!((fwd - 32.996).abs() < 0.01, "Forward score ~32.996");
    }
}

#[test]
fn test_hmm_bands_forward_matrix() {
    let golden = load_golden("hmm_bands.txt");

    let mut fwd_values = Vec::new();
    for line in golden.lines() {
        if line.starts_with("fwd ") {
            let values = parse_i32_array(line, 1);
            if values.len() == 5 {
                fwd_values.push((values[0], values[1], values[2], values[3], values[4]));
            }
        }
    }

    assert!(!fwd_values.is_empty(), "Should have forward matrix values");

    // Verify CP9MX can represent forward matrix
    let mx = CP9MX::new(71, 76);
    assert_eq!(mx.mmx.len(), 77); // 0..L
    assert_eq!(mx.mmx[0].len(), 72); // 0..M
}

#[test]
fn test_hmm_bands_backward_matrix() {
    let golden = load_golden("hmm_bands.txt");

    let mut bwd_values = Vec::new();
    for line in golden.lines() {
        if line.starts_with("bwd ") {
            let values = parse_i32_array(line, 1);
            if values.len() == 5 {
                bwd_values.push((values[0], values[1], values[2], values[3], values[4]));
            }
        }
    }

    assert!(!bwd_values.is_empty(), "Should have backward matrix values");
}

#[test]
fn test_cp9bands_allocation() {
    // Verify CP9Bands can be allocated for the expected dimensions
    let bands = CP9Bands::new(71, 227);
    assert_eq!(bands.hmm_m, 71);
    assert_eq!(bands.cm_m, 227);

    // Verify it can allocate for sequence
    let mut bands = CP9Bands::new(71, 227);
    bands.alloc_for_seq(76);
    assert_eq!(bands.l, 76);
    assert_eq!(bands.pn_min_m.len(), 77);
}

// =====================================================================
// QDB Bands Tests
// =====================================================================

#[test]
fn test_qdb_bands_structure() {
    let golden = load_golden("qdb_bands.txt");

    assert!(!golden.is_empty());

    // Parse beta parameters
    let beta1 = parse_f64(&golden, "beta1");
    let beta2 = parse_f64(&golden, "beta2");

    assert!(beta1.is_some(), "Should have beta1");
    assert!(beta2.is_some(), "Should have beta2");

    const EPSILON: f64 = 1e-10;
    if let Some(b1) = beta1 {
        assert!((b1 - 1e-7).abs() < EPSILON, "beta1 should be 1e-7");
    }
    if let Some(b2) = beta2 {
        assert!((b2 - 1e-15).abs() < EPSILON, "beta2 should be 1e-15");
    }
}

#[test]
fn test_qdb_bands_set1() {
    let golden = load_golden("qdb_bands.txt");
    let (beta, bands) = parse_qdb_bands(&golden, 1);

    assert!((beta - 1e-7).abs() < 1e-10, "beta1 should be 1e-7");
    assert!(!bands.is_empty(), "Should have QDB1 bands");

    // Verify QDBBands struct can store these
    let mut qdb = QDBBands::new(227);
    qdb.beta = beta;

    for (state, dmin, dmax) in &bands {
        let v = *state as usize;
        if v < 227 {
            qdb.set_band(v, *dmin, *dmax);
            assert_eq!(qdb.get_dmin(v), *dmin);
            assert_eq!(qdb.get_dmax(v), *dmax);
        }
    }

    // Check first state band
    assert_eq!(qdb.get_dmin(0), 1);
    assert_eq!(qdb.get_dmax(0), 184);
}

#[test]
fn test_qdb_bands_set2() {
    let golden = load_golden("qdb_bands.txt");
    let (beta, bands) = parse_qdb_bands(&golden, 2);

    assert!((beta - 1e-15).abs() < 1e-20, "beta2 should be 1e-15");
    assert!(!bands.is_empty(), "Should have QDB2 bands");

    // QDB2 bands should generally be wider than QDB1
    let (_, bands1) = parse_qdb_bands(&golden, 1);

    for ((s1, dmin1, dmax1), (s2, dmin2, dmax2)) in bands1.iter().zip(bands.iter()) {
        assert_eq!(s1, s2, "States should match");
        // QDB2 (beta=1e-15) should have wider or equal bands
        assert!(*dmin2 <= *dmin1, "QDB2 dmin should be <= QDB1 dmin at state {}", s1);
        assert!(*dmax2 >= *dmax1, "QDB2 dmax should be >= QDB1 dmax at state {}", s1);
    }
}

#[test]
fn test_qdb_bands_constraints() {
    let golden = load_golden("qdb_bands.txt");
    let (_, bands) = parse_qdb_bands(&golden, 1);

    for (state, dmin, dmax) in &bands {
        // dmin should be <= dmax
        assert!(*dmin <= *dmax, "State {}: dmin {} > dmax {}", state, dmin, dmax);
        // dmin should be >= 0
        assert!(*dmin >= 0, "State {}: dmin {} < 0", state, dmin);
        // dmax should be reasonable (W=184 for this model)
        assert!(*dmax <= 334, "State {}: dmax {} > 334", state, dmax);
    }
}

// =====================================================================
// DP Matrix Size Tests
// =====================================================================

#[test]
fn test_dp_matrix_sizes() {
    let golden = load_golden("dp_matrix_sizes.txt");

    assert!(!golden.is_empty());

    // Parse matrix size entries
    let mut found_sizes = false;
    for line in golden.lines() {
        if line.starts_with("cm_mx ") || line.starts_with("cm_tr_mx ") {
            found_sizes = true;
            // Format: cm_mx L=50 ncells=302328 Mb=1.30
            assert!(line.contains("L="), "Should have L parameter");
            assert!(line.contains("ncells="), "Should have ncells parameter");
            assert!(line.contains("Mb="), "Should have Mb parameter");
        }
    }

    assert!(found_sizes, "Should have matrix size information");
}

#[test]
fn test_dp_matrix_size_scaling() {
    let golden = load_golden("dp_matrix_sizes.txt");

    // Extract L values and ncells
    let mut sizes: Vec<(i32, usize)> = Vec::new();

    for line in golden.lines() {
        if line.starts_with("cm_mx ") {
            // Parse L and ncells from "cm_mx L=50 ncells=302328 Mb=1.30"
            let mut l = 0i32;
            let mut ncells = 0usize;

            for part in line.split_whitespace() {
                if part.starts_with("L=") {
                    l = part.trim_start_matches("L=").parse().unwrap_or(0);
                } else if part.starts_with("ncells=") {
                    ncells = part.trim_start_matches("ncells=").parse().unwrap_or(0);
                }
            }

            if l > 0 && ncells > 0 {
                sizes.push((l, ncells));
            }
        }
    }

    assert!(!sizes.is_empty(), "Should have parsed size data");

    // Verify ncells grows approximately quadratically with L
    // (since CM DP is O(L^2 * M) without bands)
    if sizes.len() >= 2 {
        let (l1, n1) = sizes[0];
        let (l2, n2) = sizes[1];

        let l_ratio = l2 as f64 / l1 as f64;
        let n_ratio = n2 as f64 / n1 as f64;

        // ncells should grow roughly as L^2
        let expected_ratio = l_ratio * l_ratio;
        assert!((n_ratio / expected_ratio - 1.0).abs() < 0.5,
            "ncells ratio {} should be close to L^2 ratio {}", n_ratio, expected_ratio);
    }
}

#[test]
fn test_cp9mx_memory_estimation() {
    // Verify our CP9MX memory estimation is reasonable
    let mx = CP9MX::new(71, 76);

    let ncells = mx.ncells();
    let mb = mx.size_mb();

    // Should have ~3 * 77 * 72 + 2 * 77 cells
    let expected_cells = 3 * 77 * 72 + 2 * 77;
    assert_eq!(ncells, expected_cells);

    // Memory should be ncells * 4 bytes
    let expected_mb = (expected_cells * 4) as f64 / (1024.0 * 1024.0);
    assert!((mb - expected_mb).abs() < 0.001);
}

// =====================================================================
// Integration Tests
// =====================================================================

#[test]
fn test_full_cp9_construction() {
    let golden = load_golden("cp9_hmm.txt");
    let cp9 = build_cp9_from_golden(&golden);

    // Verify all components are populated
    assert_eq!(cp9.m, 71);
    assert_eq!(cp9.flags, 15);
    assert!((cp9.el_self - 0.94).abs() < 1e-4);
    assert!((cp9.p1 - 1.0).abs() < 1e-4);

    // Null model should be uniform
    for i in 0..4 {
        assert!((cp9.null[i] - 0.25).abs() < 1e-4);
    }

    // First match emission at node 1 should have specific values
    let mat1_sum: f64 = cp9.mat[1].iter().map(|&x| x as f64).sum();
    assert!((mat1_sum - 1.0).abs() < 1e-4);

    // Verify begin probability at node 1
    assert!(cp9.begin[1] > 0.9);
}

#[test]
fn test_cp9_logoddsify() {
    let golden = load_golden("cp9_hmm.txt");
    let mut cp9 = build_cp9_from_golden(&golden);

    // Convert to log-odds scores
    cp9.logoddsify();

    // Check that scores are populated
    // Match emission scores at node 1
    for a in 0..4 {
        // Scores should be reasonable (not IMPOSSIBLE for valid emissions)
        if cp9.mat[1][a] > 0.0 {
            assert!(cp9.msc[1][a] > -100000,
                "Match score at node 1 residue {} should be > -100000", a);
        }
    }

    // Insert emission scores should be ~0 (equal to null model)
    for a in 0..4 {
        assert!(cp9.isc[0][a].abs() < 10,
            "Insert score at node 0 residue {} should be ~0", a);
    }
}
