//! CP9 Dynamic Programming Algorithms
//!
//! Implements Forward, Backward, Posterior Decoding and HMM band computation
//! for CP9 Profile HMMs. These algorithms are used to compute HMM bands that
//! constrain CM alignment, dramatically reducing computation.

use crate::cp9::{CP9, CP9_IMPOSSIBLE, CP9_INTSCALE, CTMM, CTMI, CTMD, CTIM, CTII, CTDM, CTDD};
use crate::cp9_bands::CP9Bands;
use crate::cp9_mx::CP9MX;

// =============================================================================
// Log-Sum-Exp Utilities for Score Arithmetic
// =============================================================================

/// Maximum value for safe log-sum-exp addition (scaled scores)
const LOG_SUM_MAX: i32 = i32::MAX / 2;

/// Safe addition of log-scaled scores, avoiding overflow
#[inline]
fn safe_score_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Add multiple scores safely (for adding 3 scores together)
#[inline]
fn safe_score_add3(a: i32, b: i32, c: i32) -> i32 {
    a.saturating_add(b).saturating_add(c)
}

/// Add two log-scaled scores using log-sum-exp: result = log(exp(a) + exp(b))
/// Scores are in scaled log-probability space (CP9_INTSCALE * log2(prob))
#[inline]
fn score_log_sum(a: i32, b: i32) -> i32 {
    if a == CP9_IMPOSSIBLE {
        return b;
    }
    if b == CP9_IMPOSSIBLE {
        return a;
    }

    // Use the larger value as base to avoid overflow
    let (max, diff) = if a > b { (a, a - b) } else { (b, b - a) };

    // If difference is too large, the smaller value contributes nothing
    if diff > (15.0 * CP9_INTSCALE as f64) as i32 {
        return max;
    }

    // log(exp(max) + exp(min)) = max + log(1 + exp(min - max))
    // = max + log(1 + exp(-diff))
    let adjustment = (1.0 + 2.0_f64.powf(-(diff as f64) / CP9_INTSCALE)).log2() * CP9_INTSCALE;
    let result = max + adjustment as i32;

    result.min(LOG_SUM_MAX)
}

// =============================================================================
// CP9 Forward Algorithm
// =============================================================================

/// Run CP9 Forward algorithm
///
/// Computes the Forward matrix for sequence dsq[1..L] using CP9 HMM.
/// Returns the Forward matrix and the total log-probability score.
///
/// # Arguments
/// * `cp9` - The CP9 profile HMM
/// * `dsq` - Digital sequence (1-indexed, dsq[1..L])
/// * `l` - Sequence length
///
/// # Returns
/// Tuple of (Forward matrix, log-probability score in scaled integer form)
pub fn cp9_forward(cp9: &CP9, dsq: &[u8], l: i32) -> (CP9MX, i32) {
    let m = cp9.m as usize;
    let l_usize = l as usize;

    let mut fwd = CP9MX::new(cp9.m, l);

    // Initialize row 0 (before sequence)
    // B state at position 0 has score 0 (probability 1)
    fwd.bmx[0] = 0;

    // M_0 is a special case (no real M_0, conceptually from B to M_1)
    // D_0 is entry to delete path
    fwd.dmx[0][0] = CP9_IMPOSSIBLE;
    fwd.imx[0][0] = CP9_IMPOSSIBLE;
    fwd.mmx[0][0] = CP9_IMPOSSIBLE;

    // Initialize first delete column: D_1..D_M
    // Can reach D_k from B via begin[k] then skip
    for k in 1..=m {
        if cp9.bsc[k] != CP9_IMPOSSIBLE {
            // Entry via begin to match, then delete transitions
            let mut dscore = cp9.bsc[k];
            // This is D_k at position 0, reachable by B->M_k->D_k? No, that needs emission
            // Actually, D states don't emit, so we can reach D_k at i=0 via B->D path
            // In plan9: D_1 can be reached from M_0 (fictional), D_k from D_{k-1}
            fwd.dmx[0][k] = CP9_IMPOSSIBLE;
        }
    }

    // Main recursion
    for i in 1..=l_usize {
        let x = dsq[i] as usize;  // Current residue (0-3 for A,C,G,U)

        // Update B state
        fwd.bmx[i] = CP9_IMPOSSIBLE;  // In local mode this could be nonzero

        // Update M, I, D for each node k
        for k in 1..=m {
            // M_k[i]: comes from M_{k-1}[i-1], I_{k-1}[i-1], D_{k-1}[i-1], or B[i-1]
            let mut m_score = CP9_IMPOSSIBLE;

            // From M_{k-1}[i-1] via M->M transition
            if k > 1 && fwd.mmx[i-1][k-1] != CP9_IMPOSSIBLE {
                let sc = fwd.mmx[i-1][k-1].saturating_add(cp9.tsc[k-1][CTMM]);
                m_score = score_log_sum(m_score, sc);
            }

            // From I_{k-1}[i-1] via I->M transition
            if k > 1 && fwd.imx[i-1][k-1] != CP9_IMPOSSIBLE {
                let sc = fwd.imx[i-1][k-1].saturating_add(cp9.tsc[k-1][CTIM]);
                m_score = score_log_sum(m_score, sc);
            }

            // From D_{k-1}[i-1] via D->M transition
            if k > 1 && fwd.dmx[i-1][k-1] != CP9_IMPOSSIBLE {
                let sc = fwd.dmx[i-1][k-1].saturating_add(cp9.tsc[k-1][CTDM]);
                m_score = score_log_sum(m_score, sc);
            }

            // From B[i-1] via begin transition (local entry)
            if fwd.bmx[i-1] != CP9_IMPOSSIBLE && cp9.bsc[k] != CP9_IMPOSSIBLE {
                let sc = fwd.bmx[i-1].saturating_add(cp9.bsc[k]);
                m_score = score_log_sum(m_score, sc);
            }

            // Add match emission score
            if m_score != CP9_IMPOSSIBLE {
                m_score = m_score.saturating_add(cp9.msc[k][x]);
            }
            fwd.mmx[i][k] = m_score;

            // I_k[i]: comes from M_k[i-1], I_k[i-1], D_k[i-1]
            let mut i_score = CP9_IMPOSSIBLE;

            // From M_k[i-1] via M->I transition
            if fwd.mmx[i-1][k] != CP9_IMPOSSIBLE {
                let sc = fwd.mmx[i-1][k].saturating_add(cp9.tsc[k][CTMI]);
                i_score = score_log_sum(i_score, sc);
            }

            // From I_k[i-1] via I->I transition
            if fwd.imx[i-1][k] != CP9_IMPOSSIBLE {
                let sc = fwd.imx[i-1][k].saturating_add(cp9.tsc[k][CTII]);
                i_score = score_log_sum(i_score, sc);
            }

            // Add insert emission score
            if i_score != CP9_IMPOSSIBLE {
                i_score = i_score.saturating_add(cp9.isc[k][x]);
            }
            fwd.imx[i][k] = i_score;

            // D_k[i]: comes from M_{k-1}[i], I_{k-1}[i], D_{k-1}[i]
            // Note: D states don't emit, so we use values at same position i
            let mut d_score = CP9_IMPOSSIBLE;

            // From M_{k-1}[i] via M->D transition
            if k > 1 && fwd.mmx[i][k-1] != CP9_IMPOSSIBLE {
                let sc = fwd.mmx[i][k-1].saturating_add(cp9.tsc[k-1][CTMD]);
                d_score = score_log_sum(d_score, sc);
            }

            // From D_{k-1}[i] via D->D transition
            if k > 1 && fwd.dmx[i][k-1] != CP9_IMPOSSIBLE {
                let sc = fwd.dmx[i][k-1].saturating_add(cp9.tsc[k-1][CTDD]);
                d_score = score_log_sum(d_score, sc);
            }

            fwd.dmx[i][k] = d_score;
        }

        // Handle I_0 (insert before first match node)
        fwd.imx[i][0] = CP9_IMPOSSIBLE;  // Simplified - no N state emissions
    }

    // Calculate final score by summing over all possible end points
    let mut final_score = CP9_IMPOSSIBLE;

    for k in 1..=m {
        // End from M_k
        if fwd.mmx[l_usize][k] != CP9_IMPOSSIBLE && cp9.esc[k] != CP9_IMPOSSIBLE {
            let sc = safe_score_add(fwd.mmx[l_usize][k], cp9.esc[k]);
            final_score = score_log_sum(final_score, sc);
        }

        // End from D_k
        if fwd.dmx[l_usize][k] != CP9_IMPOSSIBLE && cp9.esc[k] != CP9_IMPOSSIBLE {
            let sc = safe_score_add(fwd.dmx[l_usize][k], cp9.esc[k]);
            final_score = score_log_sum(final_score, sc);
        }
    }

    (fwd, final_score)
}

// =============================================================================
// CP9 Backward Algorithm
// =============================================================================

/// Run CP9 Backward algorithm
///
/// Computes the Backward matrix for sequence dsq[1..L] using CP9 HMM.
/// Returns the Backward matrix and the total log-probability score.
///
/// # Arguments
/// * `cp9` - The CP9 profile HMM
/// * `dsq` - Digital sequence (1-indexed, dsq[1..L])
/// * `l` - Sequence length
///
/// # Returns
/// Tuple of (Backward matrix, log-probability score in scaled integer form)
pub fn cp9_backward(cp9: &CP9, dsq: &[u8], l: i32) -> (CP9MX, i32) {
    let m = cp9.m as usize;
    let l_usize = l as usize;

    let mut bck = CP9MX::new(cp9.m, l);

    // Initialize row L (end of sequence)
    // All states can transition to E with their end probability
    for k in 1..=m {
        if cp9.esc[k] != CP9_IMPOSSIBLE {
            bck.mmx[l_usize][k] = cp9.esc[k];
            bck.dmx[l_usize][k] = cp9.esc[k];
        }
        bck.imx[l_usize][k] = CP9_IMPOSSIBLE;
    }

    // Main recursion (going backwards)
    for i in (0..l_usize).rev() {
        let x_next = if i < l_usize { dsq[i + 1] as usize } else { 0 };

        // Update states in reverse order: D, I, M
        for k in (1..=m).rev() {
            // D_k[i]: can go to M_{k+1}[i+1], I_k[i+1], D_{k+1}[i]
            let mut d_score = CP9_IMPOSSIBLE;

            // To M_{k+1}[i+1] via D->M transition
            if k < m && bck.mmx[i+1][k+1] != CP9_IMPOSSIBLE {
                let sc = safe_score_add3(cp9.tsc[k][CTDM], cp9.msc[k+1][x_next], bck.mmx[i+1][k+1]);
                d_score = score_log_sum(d_score, sc);
            }

            // To D_{k+1}[i] via D->D transition
            if k < m && bck.dmx[i][k+1] != CP9_IMPOSSIBLE {
                let sc = safe_score_add(cp9.tsc[k][CTDD], bck.dmx[i][k+1]);
                d_score = score_log_sum(d_score, sc);
            }

            // End probability (if at end of sequence or local mode)
            if i == l_usize && cp9.esc[k] != CP9_IMPOSSIBLE {
                d_score = score_log_sum(d_score, cp9.esc[k]);
            }

            bck.dmx[i][k] = d_score;

            // I_k[i]: can go to M_{k+1}[i+1], I_k[i+1]
            let mut i_score = CP9_IMPOSSIBLE;

            // To M_{k+1}[i+1] via I->M transition
            if k < m && i < l_usize && bck.mmx[i+1][k+1] != CP9_IMPOSSIBLE {
                let sc = safe_score_add3(cp9.tsc[k][CTIM], cp9.msc[k+1][x_next], bck.mmx[i+1][k+1]);
                i_score = score_log_sum(i_score, sc);
            }

            // To I_k[i+1] via I->I transition
            if i < l_usize && bck.imx[i+1][k] != CP9_IMPOSSIBLE {
                let sc = safe_score_add3(cp9.tsc[k][CTII], cp9.isc[k][x_next], bck.imx[i+1][k]);
                i_score = score_log_sum(i_score, sc);
            }

            bck.imx[i][k] = i_score;

            // M_k[i]: can go to M_{k+1}[i+1], I_k[i+1], D_{k+1}[i]
            let mut m_score = CP9_IMPOSSIBLE;

            // To M_{k+1}[i+1] via M->M transition
            if k < m && i < l_usize && bck.mmx[i+1][k+1] != CP9_IMPOSSIBLE {
                let sc = safe_score_add3(cp9.tsc[k][CTMM], cp9.msc[k+1][x_next], bck.mmx[i+1][k+1]);
                m_score = score_log_sum(m_score, sc);
            }

            // To I_k[i+1] via M->I transition
            if i < l_usize && bck.imx[i+1][k] != CP9_IMPOSSIBLE {
                let sc = safe_score_add3(cp9.tsc[k][CTMI], cp9.isc[k][x_next], bck.imx[i+1][k]);
                m_score = score_log_sum(m_score, sc);
            }

            // To D_{k+1}[i] via M->D transition
            if k < m && bck.dmx[i][k+1] != CP9_IMPOSSIBLE {
                let sc = safe_score_add(cp9.tsc[k][CTMD], bck.dmx[i][k+1]);
                m_score = score_log_sum(m_score, sc);
            }

            // End probability
            if i == l_usize && cp9.esc[k] != CP9_IMPOSSIBLE {
                m_score = score_log_sum(m_score, cp9.esc[k]);
            }

            bck.mmx[i][k] = m_score;
        }

        // Update B state
        bck.bmx[i] = CP9_IMPOSSIBLE;
        for k in 1..=m {
            if cp9.bsc[k] != CP9_IMPOSSIBLE && bck.mmx[i+1][k] != CP9_IMPOSSIBLE {
                let x_next = dsq[i + 1] as usize;
                let sc = safe_score_add3(cp9.bsc[k], cp9.msc[k][x_next], bck.mmx[i+1][k]);
                bck.bmx[i] = score_log_sum(bck.bmx[i], sc);
            }
        }
    }

    // Calculate final score from B[0]
    let final_score = bck.bmx[0];

    (bck, final_score)
}

// =============================================================================
// CP9 Posterior Decoding
// =============================================================================

/// Posterior probability matrix for CP9 states
#[derive(Debug, Clone)]
pub struct CP9Posterior {
    /// Model length
    pub m: i32,
    /// Sequence length
    pub l: i32,
    /// Match posteriors: pmx[i][k] = P(M_k emits x_i | sequence)
    pub pmx: Vec<Vec<f64>>,
    /// Insert posteriors: pix[i][k] = P(I_k emits x_i | sequence)
    pub pix: Vec<Vec<f64>>,
    /// Delete posteriors: pdx[i][k] = P(pass through D_k at position i | sequence)
    pub pdx: Vec<Vec<f64>>,
}

impl CP9Posterior {
    /// Create a new posterior matrix
    pub fn new(m: i32, l: i32) -> Self {
        let l_plus_1 = (l + 1) as usize;
        let m_plus_1 = (m + 1) as usize;

        CP9Posterior {
            m,
            l,
            pmx: vec![vec![0.0; m_plus_1]; l_plus_1],
            pix: vec![vec![0.0; m_plus_1]; l_plus_1],
            pdx: vec![vec![0.0; m_plus_1]; l_plus_1],
        }
    }
}

/// Compute posterior probabilities from Forward and Backward matrices
///
/// # Arguments
/// * `fwd` - Forward matrix from cp9_forward()
/// * `bck` - Backward matrix from cp9_backward()
/// * `fwd_score` - Total Forward score (log-probability)
///
/// # Returns
/// Posterior probability matrix
pub fn cp9_posterior_decode(
    fwd: &CP9MX,
    bck: &CP9MX,
    fwd_score: i32,
) -> CP9Posterior {
    let m = fwd.m as usize;
    let l = fwd.l as usize;

    let mut post = CP9Posterior::new(fwd.m, fwd.l);

    // Convert total score to probability for normalization
    // P(x) = 2^(fwd_score / INTSCALE)
    let log_total = fwd_score as f64 / CP9_INTSCALE;

    for i in 1..=l {
        for k in 1..=m {
            // Match posterior: P(M_k, i | x) = F(M_k, i) * B(M_k, i) / P(x)
            // Use i64 to avoid overflow before converting to f64
            if fwd.mmx[i][k] != CP9_IMPOSSIBLE && bck.mmx[i][k] != CP9_IMPOSSIBLE {
                let log_prob = (fwd.mmx[i][k] as i64 + bck.mmx[i][k] as i64) as f64 / CP9_INTSCALE - log_total;
                post.pmx[i][k] = 2.0_f64.powf(log_prob).min(1.0);
            }

            // Insert posterior
            if fwd.imx[i][k] != CP9_IMPOSSIBLE && bck.imx[i][k] != CP9_IMPOSSIBLE {
                let log_prob = (fwd.imx[i][k] as i64 + bck.imx[i][k] as i64) as f64 / CP9_INTSCALE - log_total;
                post.pix[i][k] = 2.0_f64.powf(log_prob).min(1.0);
            }

            // Delete posterior (delete states are at position i but don't emit)
            if fwd.dmx[i][k] != CP9_IMPOSSIBLE && bck.dmx[i][k] != CP9_IMPOSSIBLE {
                let log_prob = (fwd.dmx[i][k] as i64 + bck.dmx[i][k] as i64) as f64 / CP9_INTSCALE - log_total;
                post.pdx[i][k] = 2.0_f64.powf(log_prob).min(1.0);
            }
        }
    }

    post
}

// =============================================================================
// HMM Band Computation
// =============================================================================

/// Compute HMM bands from posterior probabilities
///
/// This function computes:
/// 1. imin[k], imax[k]: sequence position range where node k has significant probability
/// 2. pn_min_m[i], pn_max_m[i]: HMM node range at sequence position i (match states)
///
/// # Arguments
/// * `post` - Posterior probability matrix
/// * `bands` - CP9Bands structure to fill
/// * `thresh` - Posterior probability threshold (e.g., 0.01)
pub fn cp9_hmm_band_bounds(post: &CP9Posterior, bands: &mut CP9Bands, thresh: f64) {
    let m = post.m as usize;
    let l = post.l as usize;

    // Allocate bands for this sequence length
    bands.alloc_for_seq(post.l);

    // Compute imin[k], imax[k]: sequence position range for each HMM node
    for k in 1..=m {
        let mut imin = l as i32 + 1;  // Start with invalid
        let mut imax = 0i32;

        for i in 1..=l {
            // Check if this position has significant probability for this node
            let prob = post.pmx[i][k] + post.pix[i][k];
            if prob >= thresh {
                if (i as i32) < imin {
                    imin = i as i32;
                }
                if (i as i32) > imax {
                    imax = i as i32;
                }
            }
        }

        // If no positions found, use full range
        if imin > imax {
            imin = 1;
            imax = l as i32;
        }

        bands.imin[k] = imin;
        bands.imax[k] = imax;
    }

    // Compute pn_min_m[i], pn_max_m[i]: HMM node range at each sequence position
    for i in 1..=l {
        let mut kmin = (m + 1) as i32;  // Start with invalid
        let mut kmax = 0i32;

        for k in 1..=m {
            let prob = post.pmx[i][k] + post.pix[i][k];
            if prob >= thresh {
                if (k as i32) < kmin {
                    kmin = k as i32;
                }
                if (k as i32) > kmax {
                    kmax = k as i32;
                }
            }
        }

        // If no nodes found, use full range
        if kmin > kmax {
            kmin = 1;
            kmax = m as i32;
        }

        bands.pn_min_m[i] = kmin;
        bands.pn_max_m[i] = kmax;

        // Also compute insert state ranges (usually same as match)
        bands.pn_min_i[i] = kmin.saturating_sub(1).max(0);
        bands.pn_max_i[i] = (kmax + 1).min(m as i32);

        // Delete state ranges
        bands.pn_min_d[i] = kmin;
        bands.pn_max_d[i] = kmax;
    }

    // Mark bands as valid
    bands.l = post.l;
    bands.valid = true;
}

/// Main entry point: compute CP9 HMM bands for a sequence
///
/// This runs the full Forward-Backward algorithm and computes HMM bands
/// that can be used to constrain CM alignment.
///
/// # Arguments
/// * `cp9` - The CP9 profile HMM
/// * `dsq` - Digital sequence (1-indexed)
/// * `l` - Sequence length
/// * `bands` - CP9Bands structure to fill
///
/// # Returns
/// The Forward score (can be used for HMM-based filtering)
pub fn cp9_compute_bands(cp9: &CP9, dsq: &[u8], l: i32, bands: &mut CP9Bands) -> i32 {
    // Run Forward algorithm
    let (fwd, fwd_score) = cp9_forward(cp9, dsq, l);

    // Run Backward algorithm
    let (bck, _bck_score) = cp9_backward(cp9, dsq, l);

    // Compute posterior probabilities
    let post = cp9_posterior_decode(&fwd, &bck, fwd_score);

    // Compute HMM bands from posteriors
    let thresh = bands.thresh1;  // Use configured threshold
    cp9_hmm_band_bounds(&post, bands, thresh);

    fwd_score
}

// =============================================================================
// CM Band Mapping
// =============================================================================

/// Map HMM bands to CM state bands
///
/// This converts HMM position bands (which node k at position i) to
/// CM state bands (jmin, jmax, hdmin, hdmax for each state v).
///
/// # Arguments
/// * `bands` - CP9Bands with HMM bands computed
/// * `cm_m` - Number of CM states
/// * `cm2hmm` - Mapping from CM state to HMM node(s)
pub fn cp9_to_cm_bands(
    bands: &mut CP9Bands,
    cm_m: i32,
    cm2hmm: &[(i32, i32)],  // For each CM state: (left_hmm_node, right_hmm_node)
) {
    cp9_to_cm_bands_with_types(bands, cm_m, cm2hmm, None);
}

/// Enhanced CP9 to CM band mapping with proper edge case handling
///
/// When state types are provided, E_ST states get jmin=0 so d=0 is valid at all j.
/// ROOT_S and B_ST states get full sequence range, E_ST states always allow d=0.
///
/// For FULL-SEQUENCE ALIGNMENT (not search), we use relaxed bands that ensure
/// all states can be reached at all valid positions. This is necessary because:
/// 1. The HMM bands are computed for hit-detection, not full alignment
/// 2. Full alignment requires the recursion to reach ROOT_S at (j=L, d=L)
/// 3. All intermediate states must have valid bands to propagate scores up
pub fn cp9_to_cm_bands_with_types(
    bands: &mut CP9Bands,
    cm_m: i32,
    cm2hmm: &[(i32, i32)],  // For each CM state: (left_hmm_node, right_hmm_node)
    state_types: Option<&[i8]>,  // Optional: CM state types
) {
    use crate::constants::{E_ST, S_ST, B_ST, IL_ST, IR_ST, ML_ST, MR_ST, MP_ST, D_ST, EL_ST};

    let l = bands.l;

    for v in 0..(cm_m as usize) {
        // Get state type
        let st_type = state_types.map(|st| {
            if v < st.len() { st[v] as i32 } else { -1 }
        }).unwrap_or(-1);

        // Compute StateDelta (sd) - minimum d for this state type
        // From C: MP=2, ML/MR/IL/IR=1, others=0
        let sd = match st_type {
            x if x == MP_ST => 2,
            x if x == ML_ST || x == MR_ST || x == IL_ST || x == IR_ST => 1,
            _ => 0,
        };

        // Get CM state i-bands from HMM mapping
        // For left-emitting states, use left_k; for right-emitting, use right_k
        // For pair-emitting (MP), use left_k for i-band
        let (imin_v, imax_v) = if v < cm2hmm.len() {
            let (left_k, right_k) = cm2hmm[v];
            match st_type {
                // Left-emitting states: use left_k for i-band
                x if x == ML_ST || x == IL_ST => {
                    if left_k > 0 && (left_k as usize) < bands.imin.len() {
                        (bands.imin[left_k as usize], bands.imax[left_k as usize])
                    } else {
                        (0, l)
                    }
                }
                // Pair-emitting: use left_k for i-band (left emission)
                x if x == MP_ST => {
                    if left_k > 0 && (left_k as usize) < bands.imin.len() {
                        (bands.imin[left_k as usize], bands.imax[left_k as usize])
                    } else {
                        (0, l)
                    }
                }
                // Right-emitting states: i-band based on right_k (implicit)
                x if x == MR_ST || x == IR_ST => {
                    // For right-emitting states, the i position is implicitly bounded
                    // by j and the structure. Use full range but respect j constraints.
                    if right_k > 0 && (right_k as usize) < bands.imin.len() {
                        (bands.imin[right_k as usize], bands.imax[right_k as usize])
                    } else {
                        (0, l)
                    }
                }
                // Delete states: use whichever is valid
                x if x == D_ST => {
                    if left_k > 0 && (left_k as usize) < bands.imin.len() {
                        (bands.imin[left_k as usize], bands.imax[left_k as usize])
                    } else if right_k > 0 && (right_k as usize) < bands.imin.len() {
                        (bands.imin[right_k as usize], bands.imax[right_k as usize])
                    } else {
                        (0, l)
                    }
                }
                // S, B, E, EL states: full range
                _ => (0, l),
            }
        } else {
            (0, l)
        };

        // Store CM state i-bands in the bands structure
        if v < bands.imin.len() {
            bands.imin[v] = imin_v;
            bands.imax[v] = imax_v;
        }

        // For full-sequence alignment, use permissive bands for j
        // All states should be able to cover full sequence to allow recursion
        let (jmin, jmax) = match st_type {
            x if x == E_ST => {
                // E_ST: d=0 valid at any j position
                (0, l)
            }
            x if x == S_ST || x == B_ST => {
                // ROOT_S, BEGL_S, BEGR_S, and bifurcation: full range
                (0, l)
            }
            x if x == IL_ST || x == IR_ST || x == ML_ST || x == MR_ST ||
                 x == MP_ST || x == D_ST || x == EL_ST => {
                // Emit and non-emit states: use HMM-derived bands for j
                if v >= cm2hmm.len() {
                    (0, l)
                } else {
                    let (_left_k, right_k) = cm2hmm[v];
                    let jmin_v = if right_k > 0 && (right_k as usize) < bands.imin.len() {
                        bands.imin[right_k as usize].max(0)
                    } else {
                        0
                    };
                    let jmax_v = if right_k > 0 && (right_k as usize) < bands.imax.len() {
                        bands.imax[right_k as usize].min(l)
                    } else {
                        l
                    };
                    // Use actual HMM-derived bands from posteriors
                    (jmin_v, jmax_v)
                }
            }
            _ => {
                // Unknown states: use full range
                (0, l)
            }
        };

        bands.set_j_band(v, jmin, jmax);

        // Set d bands for each j position using ij2d_bands algorithm
        // From C hmmband.c:1415-1457:
        //   hdn = j - imax[v] + 1
        //   hdx = j - imin[v] + 1
        //   hdmin = max(hdn, sd)
        //   hdmax = hdx
        //   Invalid if hdx < sd
        for j in jmin..=jmax {
            // E_ST special case: always d=0
            if st_type == E_ST {
                bands.set_hd_band(v, j, 0, 0);
                continue;
            }

            // Apply ij2d_bands formula exactly as in C
            // hdn = j - imax_v + 1 (minimum d when i is at its maximum)
            // hdx = j - imin_v + 1 (maximum d when i is at its minimum)
            let hdn = j - imax_v + 1;
            let hdx = j - imin_v + 1;

            // Check for invalid bands (hdx < sd means no valid d)
            if hdx < sd {
                // Invalid band - for banded Inside, use minimal valid band
                // This allows recursion to proceed
                bands.set_hd_band(v, j, sd, sd);

                // Diagnostic logging for ROOT_S and children (states 0-4) near end of sequence
                if v <= 4 && j >= 69 && j <= 74 {
                    eprintln!("[BAND CALC] State {} j={}: INVALID hdx={} < sd={}, using ({}, {})",
                             v, j, hdx, sd, sd, sd);
                }
            } else {
                // Valid band: hdmin = max(hdn, sd), hdmax = hdx
                let hdmin = std::cmp::max(hdn, sd);
                bands.set_hd_band(v, j, hdmin, hdx);

                // Diagnostic logging for ROOT_S and children (states 0-4) near end of sequence
                if v <= 4 && j >= 69 && j <= 74 {
                    eprintln!("[BAND CALC] State {} j={}: imin={}, imax={}, hdmin={}, hdmax={}",
                             v, j, imin_v, imax_v, hdmin, hdx);
                }
            }
        }
    }

    // Debug: show imin/imax for first 10 CM states
    for v in 0..10.min(bands.cm_m) {
        eprintln!("[IBAND DEBUG] State {}: imin={}, imax={}",
                 v, bands.imin[v as usize], bands.imax[v as usize]);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cm::ALPHABET_SIZE;

    fn make_test_cp9() -> CP9 {
        let mut cp9 = CP9::new(10);

        // Set up uniform transitions
        for k in 0..=10 {
            cp9.t[k][CTMM] = 0.7;
            cp9.t[k][CTMI] = 0.15;
            cp9.t[k][CTMD] = 0.15;
            cp9.t[k][CTIM] = 0.7;
            cp9.t[k][CTII] = 0.3;
            cp9.t[k][CTDM] = 0.7;
            cp9.t[k][CTDD] = 0.3;
        }

        // Set up uniform emissions
        for k in 1..=10 {
            for a in 0..ALPHABET_SIZE {
                cp9.mat[k][a] = 0.25;
                cp9.ins[k][a] = 0.25;
            }
        }

        // Set up begin/end
        for k in 1..=10 {
            cp9.begin[k] = 0.1;
            cp9.end[k] = 0.1;
        }

        // Convert to scores
        cp9.logoddsify();

        cp9
    }

    fn make_test_dsq(len: usize) -> Vec<u8> {
        let mut dsq = vec![0u8; len + 2];
        // Fill with A, C, G, U pattern (0, 1, 2, 3)
        for i in 1..=len {
            dsq[i] = ((i - 1) % 4) as u8;
        }
        dsq
    }

    #[test]
    fn test_score_log_sum() {
        // Test with equal scores
        let result = score_log_sum(1000, 1000);
        // log(2 * exp(1000/INTSCALE)) = 1000/INTSCALE + log(2)
        // In scaled form: 1000 + INTSCALE * 1 = 2000
        assert!(result > 1000);
        assert!(result < 2100); // log(2) * 1000 ≈ 1000

        // Test with one impossible
        assert_eq!(score_log_sum(CP9_IMPOSSIBLE, 500), 500);
        assert_eq!(score_log_sum(500, CP9_IMPOSSIBLE), 500);

        // Test with both impossible
        assert_eq!(score_log_sum(CP9_IMPOSSIBLE, CP9_IMPOSSIBLE), CP9_IMPOSSIBLE);
    }

    #[test]
    fn test_cp9_forward() {
        let cp9 = make_test_cp9();
        let dsq = make_test_dsq(20);

        let (fwd, score) = cp9_forward(&cp9, dsq.as_slice(), 20);

        // Check matrix dimensions
        assert_eq!(fwd.m, 10);
        assert_eq!(fwd.l, 20);

        // Forward score should be finite
        assert!(score > CP9_IMPOSSIBLE);

        // Check that some cells were filled
        let mut nonzero = 0;
        for i in 1..=20 {
            for k in 1..=10 {
                if fwd.mmx[i][k] > CP9_IMPOSSIBLE {
                    nonzero += 1;
                }
            }
        }
        assert!(nonzero > 0, "Forward matrix should have non-impossible cells");
    }

    #[test]
    fn test_cp9_backward() {
        let cp9 = make_test_cp9();
        let dsq = make_test_dsq(20);

        let (bck, score) = cp9_backward(&cp9, dsq.as_slice(), 20);

        // Check matrix dimensions
        assert_eq!(bck.m, 10);
        assert_eq!(bck.l, 20);

        // Backward should also have some valid cells
        let mut nonzero = 0;
        for i in 1..=20 {
            for k in 1..=10 {
                if bck.mmx[i][k] > CP9_IMPOSSIBLE {
                    nonzero += 1;
                }
            }
        }
        assert!(nonzero > 0, "Backward matrix should have non-impossible cells");
    }

    #[test]
    fn test_cp9_posterior_decode() {
        let cp9 = make_test_cp9();
        let dsq = make_test_dsq(20);

        let (fwd, fwd_score) = cp9_forward(&cp9, dsq.as_slice(), 20);
        let (bck, _) = cp9_backward(&cp9, dsq.as_slice(), 20);

        let post = cp9_posterior_decode(&fwd, &bck, fwd_score);

        // Check dimensions
        assert_eq!(post.m, 10);
        assert_eq!(post.l, 20);

        // Posteriors should be in [0, 1]
        for i in 1..=20 {
            for k in 1..=10 {
                assert!(post.pmx[i][k] >= 0.0 && post.pmx[i][k] <= 1.0,
                    "Match posterior at ({}, {}) = {} out of range", i, k, post.pmx[i][k]);
            }
        }
    }

    #[test]
    fn test_cp9_hmm_band_bounds() {
        let cp9 = make_test_cp9();
        let dsq = make_test_dsq(20);

        let (fwd, fwd_score) = cp9_forward(&cp9, dsq.as_slice(), 20);
        let (bck, _) = cp9_backward(&cp9, dsq.as_slice(), 20);
        let post = cp9_posterior_decode(&fwd, &bck, fwd_score);

        let mut bands = CP9Bands::new(10, 100);
        cp9_hmm_band_bounds(&post, &mut bands, 0.01);

        // Bands should be valid
        assert!(bands.valid);
        assert_eq!(bands.l, 20);

        // Check imin/imax are reasonable
        for k in 1..=10 {
            assert!(bands.imin[k] >= 1, "imin[{}] = {} should be >= 1", k, bands.imin[k]);
            assert!(bands.imax[k] <= 20, "imax[{}] = {} should be <= 20", k, bands.imax[k]);
            assert!(bands.imin[k] <= bands.imax[k],
                "imin[{}] = {} should be <= imax[{}] = {}", k, bands.imin[k], k, bands.imax[k]);
        }

        // Check pn_min_m/pn_max_m are reasonable
        for i in 1..=20 {
            assert!(bands.pn_min_m[i] >= 1, "pn_min_m[{}] = {} should be >= 1", i, bands.pn_min_m[i]);
            assert!(bands.pn_max_m[i] <= 10, "pn_max_m[{}] = {} should be <= 10", i, bands.pn_max_m[i]);
        }
    }

    #[test]
    fn test_cp9_compute_bands() {
        let cp9 = make_test_cp9();
        let dsq = make_test_dsq(30);

        let mut bands = CP9Bands::new(10, 100);
        let score = cp9_compute_bands(&cp9, dsq.as_slice(), 30, &mut bands);

        // Should return valid score
        assert!(score > CP9_IMPOSSIBLE);

        // Bands should be valid
        assert!(bands.valid);
        assert_eq!(bands.l, 30);
    }
}
