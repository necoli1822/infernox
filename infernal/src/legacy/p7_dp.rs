//! P7 HMM Dynamic Programming algorithms
//!
//! Implements MSV, Viterbi, and Forward filters for fast sequence scanning.

use crate::p7_hmm::P7Profile;

/// MSV (Multiple Segment Viterbi) filter score
///
/// Fastest filter - ungapped diagonal scoring.
/// Returns the MSV score in nats.
pub fn p7_msv_filter(p7: &P7Profile, dsq: &[u8], l: i32) -> f32 {
    let m = p7.m as usize;
    let l_usize = l as usize;

    if l_usize == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    // MSV algorithm: ungapped local alignment (C implementation from HMMER3)
    // This is essentially Viterbi with all MM transitions = 1.0

    let nu = 2.0f32; // expected number of hits
    let tloop = ((l_usize as f32) / (l_usize as f32 + 3.0)).ln();
    let tmove = (3.0f32 / (l_usize as f32 + 3.0)).ln();
    let tbmk = (2.0f32 / ((m as f32) * (m as f32 + 1.0))).ln();
    let tej = ((nu - 1.0) / nu).ln();
    let tec = (1.0 / nu).ln();

    // DP matrices: current and previous row
    let mut mmx_prv = vec![f32::NEG_INFINITY; m + 1];
    let mut mmx_cur = vec![f32::NEG_INFINITY; m + 1];

    // Special states (indexed by position i)
    let mut xmx_n_prv = 0.0f32;
    let mut xmx_b_prv = tmove;  // S->N->B, no N-tail
    let mut xmx_e_prv = f32::NEG_INFINITY;
    let mut xmx_j_prv = f32::NEG_INFINITY;
    let mut xmx_c_prv = f32::NEG_INFINITY;

    // Main DP loop
    for i in 1..=l_usize {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;

        mmx_cur[0] = f32::NEG_INFINITY;
        let mut xmx_e_cur = f32::NEG_INFINITY;

        // Match states
        for k in 1..=m {
            if res < 4 {
                // Match emission log-odds score vs uniform background (bg=1/K=0.25).
                // p7.mat[k][res] is a probability, so the MSV score is ln(mat/bg).
                let msc = (p7.mat[k][res] / 0.25).ln();
                mmx_cur[k] = msc + mmx_prv[k - 1].max(xmx_b_prv + tbmk);
            } else {
                mmx_cur[k] = f32::NEG_INFINITY;
            }
            xmx_e_cur = xmx_e_cur.max(mmx_cur[k]);
        }

        // Special states
        let xmx_j_cur = (xmx_j_prv + tloop).max(xmx_e_cur + tej);
        let xmx_c_cur = (xmx_c_prv + tloop).max(xmx_e_cur + tec);
        let xmx_n_cur = xmx_n_prv + tloop;
        let xmx_b_cur = (xmx_n_cur + tmove).max(xmx_j_cur + tmove);

        // Swap for next iteration
        std::mem::swap(&mut mmx_prv, &mut mmx_cur);
        xmx_n_prv = xmx_n_cur;
        xmx_b_prv = xmx_b_cur;
        xmx_e_prv = xmx_e_cur;
        xmx_j_prv = xmx_j_cur;
        xmx_c_prv = xmx_c_cur;
    }

    // Return final C state score
    xmx_c_prv + tmove
}

/// Viterbi filter score
///
/// Standard HMM Viterbi with match, insert, delete states.
/// Returns the Viterbi score in nats.
pub fn p7_viterbi_filter(p7: &P7Profile, dsq: &[u8], l: i32) -> f32 {
    let m = p7.m as usize;
    let l_usize = l as usize;

    if l_usize == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    // DP matrices: M[i][k], I[i][k], D[i][k]
    // Use two rows for memory efficiency
    let mut mmx = vec![f32::NEG_INFINITY; m + 1];
    let mut imx = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx = vec![f32::NEG_INFINITY; m + 1];
    let mut mmx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut imx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx_prev = vec![f32::NEG_INFINITY; m + 1];

    // Transition indices
    const MM: usize = 0;
    const MI: usize = 1;
    const MD: usize = 2;
    const IM: usize = 3;
    const II: usize = 4;
    const DM: usize = 5;
    const DD: usize = 6;

    // Initialize: can start at any match state (local mode)
    mmx_prev[0] = 0.0;

    let mut max_score = f32::NEG_INFINITY;

    for i in 1..=l_usize {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;
        if res >= 4 { continue; }

        // Reset current row
        for k in 0..=m {
            mmx[k] = f32::NEG_INFINITY;
            imx[k] = f32::NEG_INFINITY;
            dmx[k] = f32::NEG_INFINITY;
        }

        for k in 1..=m {
            // Match state
            let emit_m = if p7.mat[k][res] > 0.0 { p7.mat[k][res].ln() } else { f32::NEG_INFINITY };

            let from_m = if mmx_prev[k-1] > f32::NEG_INFINITY && p7.trans[k-1][MM] > 0.0 {
                mmx_prev[k-1] + p7.trans[k-1][MM].ln()
            } else { f32::NEG_INFINITY };

            let from_i = if imx_prev[k-1] > f32::NEG_INFINITY && p7.trans[k-1][IM] > 0.0 {
                imx_prev[k-1] + p7.trans[k-1][IM].ln()
            } else { f32::NEG_INFINITY };

            let from_d = if dmx_prev[k-1] > f32::NEG_INFINITY && p7.trans[k-1][DM] > 0.0 {
                dmx_prev[k-1] + p7.trans[k-1][DM].ln()
            } else { f32::NEG_INFINITY };

            // Local entry: can start fresh at any match state
            let local_entry = 0.0;

            mmx[k] = emit_m + from_m.max(from_i).max(from_d).max(local_entry);

            // Insert state
            let emit_i = if p7.ins[k][res] > 0.0 { p7.ins[k][res].ln() } else { f32::NEG_INFINITY };

            let from_m_to_i = if mmx_prev[k] > f32::NEG_INFINITY && p7.trans[k][MI] > 0.0 {
                mmx_prev[k] + p7.trans[k][MI].ln()
            } else { f32::NEG_INFINITY };

            let from_i_to_i = if imx_prev[k] > f32::NEG_INFINITY && p7.trans[k][II] > 0.0 {
                imx_prev[k] + p7.trans[k][II].ln()
            } else { f32::NEG_INFINITY };

            imx[k] = emit_i + from_m_to_i.max(from_i_to_i);

            // Delete state (no emission)
            let from_m_to_d = if mmx[k-1] > f32::NEG_INFINITY && p7.trans[k-1][MD] > 0.0 {
                mmx[k-1] + p7.trans[k-1][MD].ln()
            } else { f32::NEG_INFINITY };

            let from_d_to_d = if dmx[k-1] > f32::NEG_INFINITY && p7.trans[k-1][DD] > 0.0 {
                dmx[k-1] + p7.trans[k-1][DD].ln()
            } else { f32::NEG_INFINITY };

            dmx[k] = from_m_to_d.max(from_d_to_d);

            // Track max score (local exit from any match state)
            if mmx[k] > max_score {
                max_score = mmx[k];
            }
        }

        // Swap rows
        std::mem::swap(&mut mmx, &mut mmx_prev);
        std::mem::swap(&mut imx, &mut imx_prev);
        std::mem::swap(&mut dmx, &mut dmx_prev);
    }

    max_score
}

/// Forward filter score
///
/// Full Forward algorithm summing over all paths.
/// Returns the Forward score in nats.
pub fn p7_forward_filter(p7: &P7Profile, dsq: &[u8], l: i32) -> f32 {
    let m = p7.m as usize;
    let l_usize = l as usize;

    if l_usize == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    // DP matrices using log-sum-exp for numerical stability
    let mut mmx = vec![f32::NEG_INFINITY; m + 1];
    let mut imx = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx = vec![f32::NEG_INFINITY; m + 1];
    let mut mmx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut imx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx_prev = vec![f32::NEG_INFINITY; m + 1];

    const MM: usize = 0;
    const MI: usize = 1;
    const MD: usize = 2;
    const IM: usize = 3;
    const II: usize = 4;
    const DM: usize = 5;
    const DD: usize = 6;

    // Initialize
    mmx_prev[0] = 0.0;

    for i in 1..=l_usize {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;
        if res >= 4 { continue; }

        for k in 0..=m {
            mmx[k] = f32::NEG_INFINITY;
            imx[k] = f32::NEG_INFINITY;
            dmx[k] = f32::NEG_INFINITY;
        }

        for k in 1..=m {
            let emit_m = if p7.mat[k][res] > 0.0 { p7.mat[k][res].ln() } else { f32::NEG_INFINITY };
            let emit_i = if p7.ins[k][res] > 0.0 { p7.ins[k][res].ln() } else { f32::NEG_INFINITY };

            // Match: log-sum-exp of incoming paths
            let mut m_scores = Vec::new();

            if mmx_prev[k-1] > f32::NEG_INFINITY && p7.trans[k-1][MM] > 0.0 {
                m_scores.push(mmx_prev[k-1] + p7.trans[k-1][MM].ln());
            }
            if imx_prev[k-1] > f32::NEG_INFINITY && p7.trans[k-1][IM] > 0.0 {
                m_scores.push(imx_prev[k-1] + p7.trans[k-1][IM].ln());
            }
            if dmx_prev[k-1] > f32::NEG_INFINITY && p7.trans[k-1][DM] > 0.0 {
                m_scores.push(dmx_prev[k-1] + p7.trans[k-1][DM].ln());
            }
            m_scores.push(0.0); // Local entry

            mmx[k] = emit_m + log_sum_exp(&m_scores);

            // Insert
            let mut i_scores = Vec::new();
            if mmx_prev[k] > f32::NEG_INFINITY && p7.trans[k][MI] > 0.0 {
                i_scores.push(mmx_prev[k] + p7.trans[k][MI].ln());
            }
            if imx_prev[k] > f32::NEG_INFINITY && p7.trans[k][II] > 0.0 {
                i_scores.push(imx_prev[k] + p7.trans[k][II].ln());
            }

            if !i_scores.is_empty() {
                imx[k] = emit_i + log_sum_exp(&i_scores);
            }

            // Delete
            let mut d_scores = Vec::new();
            if mmx[k-1] > f32::NEG_INFINITY && p7.trans[k-1][MD] > 0.0 {
                d_scores.push(mmx[k-1] + p7.trans[k-1][MD].ln());
            }
            if dmx[k-1] > f32::NEG_INFINITY && p7.trans[k-1][DD] > 0.0 {
                d_scores.push(dmx[k-1] + p7.trans[k-1][DD].ln());
            }

            if !d_scores.is_empty() {
                dmx[k] = log_sum_exp(&d_scores);
            }
        }

        std::mem::swap(&mut mmx, &mut mmx_prev);
        std::mem::swap(&mut imx, &mut imx_prev);
        std::mem::swap(&mut dmx, &mut dmx_prev);
    }

    // Sum over all exit points (local exit from any match state)
    let mut exit_scores = Vec::new();
    for k in 1..=m {
        if mmx_prev[k] > f32::NEG_INFINITY {
            exit_scores.push(mmx_prev[k]);
        }
    }

    if exit_scores.is_empty() {
        f32::NEG_INFINITY
    } else {
        log_sum_exp(&exit_scores)
    }
}

/// Log-sum-exp for numerical stability
fn log_sum_exp(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return f32::NEG_INFINITY;
    }

    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_score == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }

    let sum: f32 = scores.iter()
        .filter(|&&s| s > f32::NEG_INFINITY)
        .map(|&s| (s - max_score).exp())
        .sum();

    max_score + sum.ln()
}

/// Convert nat score to bit score
pub fn nats_to_bits(nats: f32) -> f32 {
    nats / std::f32::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p7_hmm::P7Profile;

    #[test]
    fn test_p7_msv_basic() {
        let mut p7 = P7Profile::new(5);
        // MSV works with log probabilities, but resets when score goes negative
        // To accumulate scores, we need emissions that keep score positive
        // This is a bit artificial but demonstrates the algorithm
        for k in 1..=5 {
            p7.mat[k] = [0.7, 0.1, 0.1, 0.1]; // Prefer A
        }

        // Sequence AAAAA
        let dsq = vec![255, 0, 0, 0, 0, 0, 255]; // sentinels + AAAAA
        let score = p7_msv_filter(&p7, &dsq, 5);

        // MSV should return a finite score
        assert!(score.is_finite(), "MSV score should be finite");

        // MSV aligns the whole ungapped diagonal (5 matches of A, each with
        // log-odds ln(0.7/0.25)) minus the entry/exit/length-model costs. The
        // resulting C-state score is ≈ -0.2148 bits (matches p7_simd::p7_msv_scalar).
        let expected = -0.214759f32;
        assert!((score - expected).abs() < 0.001,
            "MSV score should be approximately {}, got {}", expected, score);
    }

    #[test]
    fn test_p7_viterbi_basic() {
        let mut p7 = P7Profile::new(3);
        for k in 1..=3 {
            p7.mat[k] = [0.7, 0.1, 0.1, 0.1];
            p7.trans[k-1][0] = 0.9; // MM transition
        }

        let dsq = vec![255, 0, 0, 0, 255];
        let score = p7_viterbi_filter(&p7, &dsq, 3);
        assert!(score > f32::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp() {
        let scores = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&scores);
        // Should be approximately ln(e^1 + e^2 + e^3) ≈ 3.41
        assert!((result - 3.41).abs() < 0.1);
    }
}
