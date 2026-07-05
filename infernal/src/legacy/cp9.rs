//! CP9 Profile HMM - 1:1 port from cp9.h
//!
//! CP9 is a "Cove-Plan9" profile HMM built from a CM. It is used for
//! efficient banded alignment using the CP9 forward/backward algorithms.

use crate::cm::ALPHABET_SIZE;

// =============================================================================
// CP9 Transition Indices
// =============================================================================

/// Number of transitions per node in CP9
pub const CP9_NTRANS: usize = 10;

// CP9 transition type indices (matches enum cp9_tsc_idx_e)
/// Match to Match
pub const CTMM: usize = 0;
/// Match to Insert
pub const CTMI: usize = 1;
/// Match to Delete
pub const CTMD: usize = 2;
/// Match to EL (local end)
pub const CTMEL: usize = 3;
/// Insert to Match
pub const CTIM: usize = 4;
/// Insert to Insert
pub const CTII: usize = 5;
/// Insert to Delete (not used, always 0)
pub const CTID: usize = 6;
/// Delete to Match
pub const CTDM: usize = 7;
/// Delete to Insert (not used, always 0)
pub const CTDI: usize = 8;
/// Delete to Delete
pub const CTDD: usize = 9;

// =============================================================================
// CP9 Flags
// =============================================================================

/// CP9 has valid match scores
pub const CP9_HASBITS: u32 = 1 << 0;
/// CP9 has valid transitions
pub const CP9_HASTRANS: u32 = 1 << 1;
/// CP9 has null model
pub const CP9_HASNULL: u32 = 1 << 2;
/// CP9 is in local mode
pub const CP9_LOCAL: u32 = 1 << 3;
/// CP9 has EL states
pub const CP9_EL: u32 = 1 << 4;

// =============================================================================
// CP9 Score Constants
// =============================================================================

/// Impossible score for CP9 (integer scaled)
pub const CP9_IMPOSSIBLE: i32 = -987654321;

/// Scale factor for converting log probabilities to integer scores
pub const CP9_INTSCALE: f64 = 1000.0;

// =============================================================================
// CP9 HMM Structure
// =============================================================================

/// CP9 Profile HMM
///
/// A plan-9 profile HMM built from a covariance model.
/// Used for computing HMM bands to accelerate CM alignment.
#[derive(Debug, Clone)]
pub struct CP9 {
    // =========================================================================
    // Model dimensions
    // =========================================================================
    /// Number of match nodes (model length)
    pub m: i32,

    /// Configuration flags
    pub flags: u32,

    // =========================================================================
    // Special state parameters
    // =========================================================================
    /// EL self-transition probability
    pub el_self: f32,

    /// p1 = loop probability for N, C, J states
    pub p1: f32,

    // =========================================================================
    // Null model
    // =========================================================================
    /// Background/null model probabilities [A, C, G, U]
    pub null: [f32; ALPHABET_SIZE],

    // =========================================================================
    // Transition probabilities (probability space)
    // =========================================================================
    /// Transition probabilities [0..M][0..9]
    /// t[k][CTMM] = M_k -> M_k+1 transition probability
    pub t: Vec<[f32; CP9_NTRANS]>,

    // =========================================================================
    // Emission probabilities (probability space)
    // =========================================================================
    /// Match emission probabilities [1..M][A,C,G,U]
    /// mat[k][a] = P(emit a | M_k)
    pub mat: Vec<[f32; ALPHABET_SIZE]>,

    /// Insert emission probabilities [0..M][A,C,G,U]
    /// ins[k][a] = P(emit a | I_k)
    pub ins: Vec<[f32; ALPHABET_SIZE]>,

    // =========================================================================
    // Begin/End probabilities
    // =========================================================================
    /// Begin probabilities [1..M]
    /// begin[k] = probability of starting at M_k
    pub begin: Vec<f32>,

    /// End probabilities [1..M]
    /// end[k] = probability of ending at M_k
    pub end: Vec<f32>,

    // =========================================================================
    // Integer log-odds scores (scaled)
    // =========================================================================
    /// Transition scores [0..M][0..9] (integer scaled)
    pub tsc: Vec<[i32; CP9_NTRANS]>,

    /// Match emission scores [0..M][A,C,G,U] (integer scaled)
    pub msc: Vec<[i32; ALPHABET_SIZE]>,

    /// Insert emission scores [0..M][A,C,G,U] (integer scaled)
    pub isc: Vec<[i32; ALPHABET_SIZE]>,

    /// Begin scores [0..M] (integer scaled)
    pub bsc: Vec<i32>,

    /// End scores [0..M] (integer scaled)
    pub esc: Vec<i32>,

    // =========================================================================
    // EL state scores
    // =========================================================================
    /// EL self-transition score (integer scaled)
    pub el_selfsc: i32,

    /// Has EL state at this position [0..M]
    pub has_el: Vec<bool>,

    /// EL emission scores [0..M][A,C,G,U]
    pub el_sc: Vec<[i32; ALPHABET_SIZE]>,

    // =========================================================================
    // Special state scores (N, B, E, C, J)
    // =========================================================================
    /// Other score components (for N, C, J loops)
    pub otsc: Vec<i32>,
}

impl CP9 {
    /// Create a new CP9 HMM with the given model length
    pub fn new(m: i32) -> Self {
        let m_plus_1 = (m + 1) as usize;
        let m_plus_2 = (m + 2) as usize;

        CP9 {
            m,
            flags: 0,
            el_self: 0.0,
            p1: 1.0,
            null: [0.25; ALPHABET_SIZE],
            t: vec![[0.0; CP9_NTRANS]; m_plus_2],
            mat: vec![[0.0; ALPHABET_SIZE]; m_plus_1],
            ins: vec![[0.0; ALPHABET_SIZE]; m_plus_1],
            begin: vec![0.0; m_plus_1],
            end: vec![0.0; m_plus_1],
            tsc: vec![[CP9_IMPOSSIBLE; CP9_NTRANS]; m_plus_2],
            msc: vec![[CP9_IMPOSSIBLE; ALPHABET_SIZE]; m_plus_1],
            isc: vec![[CP9_IMPOSSIBLE; ALPHABET_SIZE]; m_plus_1],
            bsc: vec![CP9_IMPOSSIBLE; m_plus_1],
            esc: vec![CP9_IMPOSSIBLE; m_plus_1],
            el_selfsc: CP9_IMPOSSIBLE,
            has_el: vec![false; m_plus_1],
            el_sc: vec![[CP9_IMPOSSIBLE; ALPHABET_SIZE]; m_plus_1],
            otsc: vec![0; 8], // Space for N, C, J state scores
        }
    }

    /// Check if CP9 is configured in local mode
    pub fn is_local(&self) -> bool {
        (self.flags & CP9_LOCAL) != 0
    }

    /// Check if CP9 has EL states enabled
    pub fn has_el_states(&self) -> bool {
        (self.flags & CP9_EL) != 0
    }

    /// Get transition probability from node k
    pub fn get_t(&self, k: usize, trans_type: usize) -> f32 {
        self.t[k][trans_type]
    }

    /// Get match emission probability at node k for residue a
    pub fn get_mat(&self, k: usize, a: usize) -> f32 {
        self.mat[k][a]
    }

    /// Get insert emission probability at node k for residue a
    pub fn get_ins(&self, k: usize, a: usize) -> f32 {
        self.ins[k][a]
    }

    /// Get transition score from node k (integer scaled)
    pub fn get_tsc(&self, k: usize, trans_type: usize) -> i32 {
        self.tsc[k][trans_type]
    }

    /// Get match emission score at node k for residue a (integer scaled)
    pub fn get_msc(&self, k: usize, a: usize) -> i32 {
        self.msc[k][a]
    }

    /// Get insert emission score at node k for residue a (integer scaled)
    pub fn get_isc(&self, k: usize, a: usize) -> i32 {
        self.isc[k][a]
    }

    /// Convert probability to integer log-odds score
    /// score = INTSCALE * log2(prob / null)
    pub fn prob_to_score(prob: f64, null: f64) -> i32 {
        if prob <= 0.0 {
            CP9_IMPOSSIBLE
        } else {
            let log_odds = (prob / null).ln() / std::f64::consts::LN_2;
            (CP9_INTSCALE * log_odds).round() as i32
        }
    }

    /// Convert integer log-odds score to probability
    pub fn score_to_prob(score: i32, null: f64) -> f64 {
        if score <= CP9_IMPOSSIBLE / 2 {
            0.0
        } else {
            let log_odds = score as f64 / CP9_INTSCALE;
            null * 2.0_f64.powf(log_odds)
        }
    }

    /// Compute integer-scaled transition scores from probabilities
    /// Score = INTSCALE * log2(prob)
    pub fn compute_tsc(&mut self) {
        let m = self.m as usize;

        // Transition scores: tsc[k][trans] = scaled log2(prob)
        for k in 0..=m + 1 {
            for t in 0..CP9_NTRANS {
                let prob = self.t[k][t] as f64;
                if prob > 0.0 {
                    self.tsc[k][t] = (CP9_INTSCALE * prob.ln() / std::f64::consts::LN_2).round() as i32;
                } else {
                    self.tsc[k][t] = CP9_IMPOSSIBLE;
                }
            }
        }
    }

    /// Compute integer-scaled match emission scores
    /// Score = INTSCALE * log2(prob / null)
    pub fn compute_msc(&mut self) {
        let m = self.m as usize;

        // Match emission scores: msc[k][residue] = scaled log-odds
        for k in 1..=m {
            for a in 0..ALPHABET_SIZE {
                let prob = self.mat[k][a] as f64;
                let null = self.null[a] as f64;
                self.msc[k][a] = Self::prob_to_score(prob, null);
            }
        }
    }

    /// Compute integer-scaled insert emission scores
    /// Score = INTSCALE * log2(prob / null)
    pub fn compute_isc(&mut self) {
        let m = self.m as usize;

        // Insert emission scores: isc[k][residue] = scaled log-odds
        for k in 0..=m {
            for a in 0..ALPHABET_SIZE {
                let prob = self.ins[k][a] as f64;
                let null = self.null[a] as f64;
                self.isc[k][a] = Self::prob_to_score(prob, null);
            }
        }
    }

    /// Compute integer-scaled begin/end scores
    /// Score = INTSCALE * log2(prob)
    pub fn compute_bsc_esc(&mut self) {
        let m = self.m as usize;

        // Begin scores
        for k in 1..=m {
            let prob = self.begin[k] as f64;
            if prob > 0.0 {
                self.bsc[k] = (CP9_INTSCALE * prob.ln() / std::f64::consts::LN_2).round() as i32;
            } else {
                self.bsc[k] = CP9_IMPOSSIBLE;
            }
        }

        // End scores
        for k in 1..=m {
            let prob = self.end[k] as f64;
            if prob > 0.0 {
                self.esc[k] = (CP9_INTSCALE * prob.ln() / std::f64::consts::LN_2).round() as i32;
            } else {
                self.esc[k] = CP9_IMPOSSIBLE;
            }
        }

        // EL self-transition score
        if self.el_self > 0.0 {
            self.el_selfsc = (CP9_INTSCALE * (self.el_self as f64).ln() / std::f64::consts::LN_2).round() as i32;
        }
    }

    /// Configure CP9 scores from probabilities
    pub fn logoddsify(&mut self) {
        self.compute_tsc();
        self.compute_msc();
        self.compute_isc();
        self.compute_bsc_esc();

        self.flags |= CP9_HASBITS | CP9_HASTRANS;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cp9_new() {
        let cp9 = CP9::new(71);
        assert_eq!(cp9.m, 71);
        assert_eq!(cp9.t.len(), 73); // 0..M+1
        assert_eq!(cp9.mat.len(), 72); // 0..M
        assert_eq!(cp9.ins.len(), 72);
        assert_eq!(cp9.begin.len(), 72);
        assert_eq!(cp9.end.len(), 72);
    }

    #[test]
    fn test_cp9_null_model() {
        let cp9 = CP9::new(10);
        for i in 0..ALPHABET_SIZE {
            assert!((cp9.null[i] - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_prob_to_score() {
        // Probability = null => score = 0
        let score = CP9::prob_to_score(0.25, 0.25);
        assert!(score.abs() < 10); // Allow small rounding error

        // Probability = 0 => impossible score
        let score = CP9::prob_to_score(0.0, 0.25);
        assert_eq!(score, CP9_IMPOSSIBLE);

        // Probability > null => positive score
        let score = CP9::prob_to_score(0.5, 0.25);
        assert!(score > 0);

        // Probability < null => negative score
        let score = CP9::prob_to_score(0.125, 0.25);
        assert!(score < 0);
    }

    #[test]
    fn test_score_to_prob() {
        // Score = 0 => probability = null
        let prob = CP9::score_to_prob(0, 0.25);
        assert!((prob - 0.25).abs() < 1e-6);

        // Impossible score => probability = 0
        let prob = CP9::score_to_prob(CP9_IMPOSSIBLE, 0.25);
        assert!((prob - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_transition_indices() {
        assert_eq!(CTMM, 0);
        assert_eq!(CTMI, 1);
        assert_eq!(CTMD, 2);
        assert_eq!(CTMEL, 3);
        assert_eq!(CTIM, 4);
        assert_eq!(CTII, 5);
        assert_eq!(CTID, 6);
        assert_eq!(CTDM, 7);
        assert_eq!(CTDI, 8);
        assert_eq!(CTDD, 9);
    }

    #[test]
    fn test_cp9_flags() {
        let mut cp9 = CP9::new(10);
        assert!(!cp9.is_local());
        assert!(!cp9.has_el_states());

        cp9.flags |= CP9_LOCAL;
        assert!(cp9.is_local());

        cp9.flags |= CP9_EL;
        assert!(cp9.has_el_states());
    }
}
