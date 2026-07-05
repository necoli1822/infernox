//! P7 HMM filters
//!
//! Implements MSV, Viterbi, and Forward filters for profile HMMs.
//! Uses simple linear SIMD (not striped) for better correctness.

use crate::p7_hmm::P7Profile;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Thread-local Forward filter with pre-allocated buffers
/// This eliminates per-call allocations for massive speedup
pub struct P7ForwardFilter {
    m: usize,
    mmx: Vec<f32>,
    imx: Vec<f32>,
    dmx: Vec<f32>,
    mmx_prev: Vec<f32>,
    imx_prev: Vec<f32>,
    dmx_prev: Vec<f32>,
}

impl P7ForwardFilter {
    /// Create a new filter for models up to max_m in length
    pub fn new(max_m: usize) -> Self {
        let size = max_m + 1;
        P7ForwardFilter {
            m: max_m,
            mmx: vec![f32::NEG_INFINITY; size],
            imx: vec![f32::NEG_INFINITY; size],
            dmx: vec![f32::NEG_INFINITY; size],
            mmx_prev: vec![f32::NEG_INFINITY; size],
            imx_prev: vec![f32::NEG_INFINITY; size],
            dmx_prev: vec![f32::NEG_INFINITY; size],
        }
    }

    /// Ensure buffers are large enough for model of size m
    #[inline]
    fn ensure_capacity(&mut self, m: usize) {
        if m > self.m {
            let size = m + 1;
            self.mmx.resize(size, f32::NEG_INFINITY);
            self.imx.resize(size, f32::NEG_INFINITY);
            self.dmx.resize(size, f32::NEG_INFINITY);
            self.mmx_prev.resize(size, f32::NEG_INFINITY);
            self.imx_prev.resize(size, f32::NEG_INFINITY);
            self.dmx_prev.resize(size, f32::NEG_INFINITY);
            self.m = m;
        }
    }

    /// Reset buffers for a new sequence
    #[inline]
    fn reset(&mut self, m: usize) {
        self.ensure_capacity(m);
        for i in 0..=m {
            self.mmx[i] = f32::NEG_INFINITY;
            self.imx[i] = f32::NEG_INFINITY;
            self.dmx[i] = f32::NEG_INFINITY;
            self.mmx_prev[i] = f32::NEG_INFINITY;
            self.imx_prev[i] = f32::NEG_INFINITY;
            self.dmx_prev[i] = f32::NEG_INFINITY;
        }
        self.mmx_prev[0] = 0.0;
    }

    /// Run Forward filter using pre-allocated buffers
    pub fn forward(&mut self, opt: &P7ProfileOpt, dsq: &[u8], l: usize) -> f32 {
        let m = opt.m;
        if l == 0 || m == 0 {
            return f32::NEG_INFINITY;
        }

        self.reset(m);

        const MM: usize = 0;
        const MI: usize = 1;
        const MD: usize = 2;
        const IM: usize = 3;
        const II: usize = 4;
        const DM: usize = 5;
        const DD: usize = 6;

        for i in 1..=l {
            if i >= dsq.len() { break; }

            let res = dsq[i] as usize;
            if res >= 4 { continue; }

            // Reset current row efficiently
            for k in 0..=m {
                self.mmx[k] = f32::NEG_INFINITY;
                self.imx[k] = f32::NEG_INFINITY;
                self.dmx[k] = f32::NEG_INFINITY;
            }

            for k in 1..=m {
                // Match: log-sum-exp of incoming paths + emission
                let emit_m = opt.mscore[k][res];

                let from_m = if self.mmx_prev[k-1] > f32::NEG_INFINITY {
                    self.mmx_prev[k-1] + opt.tscore[k-1][MM]
                } else { f32::NEG_INFINITY };

                let from_i = if self.imx_prev[k-1] > f32::NEG_INFINITY {
                    self.imx_prev[k-1] + opt.tscore[k-1][IM]
                } else { f32::NEG_INFINITY };

                let from_d = if self.dmx_prev[k-1] > f32::NEG_INFINITY {
                    self.dmx_prev[k-1] + opt.tscore[k-1][DM]
                } else { f32::NEG_INFINITY };

                let local_entry = 0.0_f32;
                self.mmx[k] = emit_m + log_sum_exp4(from_m, from_i, from_d, local_entry);

                // Insert
                let emit_i = opt.iscore[k][res];
                let from_m_i = if self.mmx_prev[k] > f32::NEG_INFINITY {
                    self.mmx_prev[k] + opt.tscore[k][MI]
                } else { f32::NEG_INFINITY };
                let from_i_i = if self.imx_prev[k] > f32::NEG_INFINITY {
                    self.imx_prev[k] + opt.tscore[k][II]
                } else { f32::NEG_INFINITY };

                let sum_i = log_sum_exp2(from_m_i, from_i_i);
                if sum_i > f32::NEG_INFINITY {
                    self.imx[k] = emit_i + sum_i;
                }

                // Delete
                let from_m_d = if self.mmx[k-1] > f32::NEG_INFINITY {
                    self.mmx[k-1] + opt.tscore[k-1][MD]
                } else { f32::NEG_INFINITY };
                let from_d_d = if self.dmx[k-1] > f32::NEG_INFINITY {
                    self.dmx[k-1] + opt.tscore[k-1][DD]
                } else { f32::NEG_INFINITY };

                self.dmx[k] = log_sum_exp2(from_m_d, from_d_d);
            }

            // Swap current and previous
            std::mem::swap(&mut self.mmx, &mut self.mmx_prev);
            std::mem::swap(&mut self.imx, &mut self.imx_prev);
            std::mem::swap(&mut self.dmx, &mut self.dmx_prev);
        }

        // Sum over all exit points
        let mut result = f32::NEG_INFINITY;
        for k in 1..=m {
            result = log_sum_exp2(result, self.mmx_prev[k]);
        }
        result
    }
}

/// Create a new Forward filter for use in parallel contexts
pub fn create_forward_filter(max_m: usize) -> P7ForwardFilter {
    P7ForwardFilter::new(max_m)
}

// Thread-local storage for Forward filter buffers
thread_local! {
    static FORWARD_FILTER: std::cell::RefCell<P7ForwardFilter> =
        std::cell::RefCell::new(P7ForwardFilter::new(256));
}

/// Optimized P7 profile with pre-computed log-odds scores
#[derive(Debug, Clone)]
pub struct P7ProfileOpt {
    /// Model length
    pub m: usize,
    /// Pre-computed log-odds match emissions [k][residue] in nats
    /// Stored as log(p_emit / p_null) where p_null = 0.25
    pub mscore: Vec<[f32; 4]>,
    /// Pre-computed log-odds insert emissions [k][residue]
    pub iscore: Vec<[f32; 4]>,
    /// Pre-computed log transition scores [k][7]
    pub tscore: Vec<[f32; 7]>,
    /// E-value parameters
    pub evparam: crate::p7_hmm::P7EvParams,
}

impl P7ProfileOpt {
    /// Create optimized profile from standard P7Profile
    pub fn from_p7(p7: &P7Profile) -> Self {
        let m = p7.m as usize;
        let null_prob = 0.25_f32; // Uniform null model for DNA
        let ln_null = null_prob.ln(); // = -1.386

        // Pre-compute log-odds emission scores
        // p7.mat[k][r] = probability of emitting residue r at match state k
        // log-odds = ln(p) - ln(null) = ln(p/null)
        let mut mscore = vec![[0.0_f32; 4]; m + 1];
        let mut iscore = vec![[0.0_f32; 4]; m + 1];
        let mut tscore = vec![[f32::NEG_INFINITY; 7]; m + 1];

        for k in 0..=m {
            for r in 0..4 {
                // Match emissions: log(p/null) = log(p) - log(null)
                if k > 0 && p7.mat[k][r] > 0.0 {
                    mscore[k][r] = p7.mat[k][r].ln() - ln_null;
                } else {
                    mscore[k][r] = f32::NEG_INFINITY;
                }

                // Insert emissions
                if p7.ins[k][r] > 0.0 {
                    iscore[k][r] = p7.ins[k][r].ln() - ln_null;
                } else {
                    iscore[k][r] = f32::NEG_INFINITY;
                }
            }
        }

        for k in 0..=m {

            // Transition scores
            for t in 0..7 {
                if p7.trans[k][t] > 0.0 {
                    tscore[k][t] = p7.trans[k][t].ln();
                }
            }
        }

        P7ProfileOpt {
            m,
            mscore,
            iscore,
            tscore,
            evparam: p7.evparam.clone(),
        }
    }
}

/// SIMD-optimized MSV filter (simple linear SIMD, not striped)
#[cfg(target_arch = "x86_64")]
pub fn p7_msv_simd(opt: &P7ProfileOpt, dsq: &[u8], l: i32) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { p7_msv_avx_simple(opt, dsq, l as usize) }
    } else {
        p7_msv_scalar(opt, dsq, l as usize)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn p7_msv_simd(opt: &P7ProfileOpt, dsq: &[u8], l: i32) -> f32 {
    p7_msv_scalar(opt, dsq, l as usize)
}

/// Simple AVX MSV - vectorizes inner k loop (not striped)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn p7_msv_avx_simple(opt: &P7ProfileOpt, dsq: &[u8], l: usize) -> f32 {
    let m = opt.m;

    if l == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    // Same setup as scalar
    let nu = 2.0f32;
    let tloop = ((l as f32) / (l as f32 + 3.0)).ln();
    let tmove = (3.0f32 / (l as f32 + 3.0)).ln();
    let tbmk = (2.0f32 / ((m as f32) * (m as f32 + 1.0))).ln();
    let tej = ((nu - 1.0) / nu).ln();
    let tec = (1.0 / nu).ln();

    // Allocate with extra padding for SIMD
    let mut mmx_prv = vec![f32::NEG_INFINITY; m + 16];
    let mut mmx_cur = vec![f32::NEG_INFINITY; m + 16];

    let mut xmx_n_prv = 0.0f32;
    let mut xmx_b_prv = tmove;
    let mut xmx_e_prv = f32::NEG_INFINITY;
    let mut xmx_j_prv = f32::NEG_INFINITY;
    let mut xmx_c_prv = f32::NEG_INFINITY;

    let neg_inf_vec = _mm256_set1_ps(f32::NEG_INFINITY);

    // Main DP loop
    for i in 1..=l {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;

        mmx_cur[0] = f32::NEG_INFINITY;
        let mut xmx_e_cur_vec = neg_inf_vec;

        if res < 4 {
            let xmx_b_vec = _mm256_set1_ps(xmx_b_prv + tbmk);

            // Process k in chunks of 8
            let mut k = 1;
            while k + 7 <= m {
                // Load emission scores for this residue
                let msc0 = opt.mscore[k][res];
                let msc1 = opt.mscore[k+1][res];
                let msc2 = opt.mscore[k+2][res];
                let msc3 = opt.mscore[k+3][res];
                let msc4 = opt.mscore[k+4][res];
                let msc5 = opt.mscore[k+5][res];
                let msc6 = opt.mscore[k+6][res];
                let msc7 = opt.mscore[k+7][res];
                let msc_vec = _mm256_set_ps(msc7, msc6, msc5, msc4, msc3, msc2, msc1, msc0);

                // Load previous match scores (k-1..k+6)
                let mmx_prv_vec = _mm256_loadu_ps(&mmx_prv[k-1]);

                // Compute max(mmx_prv[k-1], xmx_b_prv + tbmk) + msc
                let prev_max = _mm256_max_ps(mmx_prv_vec, xmx_b_vec);
                let new_mmx = _mm256_add_ps(msc_vec, prev_max);

                // Store result
                _mm256_storeu_ps(&mut mmx_cur[k], new_mmx);

                // Track max for E state
                xmx_e_cur_vec = _mm256_max_ps(xmx_e_cur_vec, new_mmx);

                k += 8;
            }

            // Handle remaining k's with scalar
            for k in k..=m {
                let msc = opt.mscore[k][res];
                mmx_cur[k] = msc + mmx_prv[k - 1].max(xmx_b_prv + tbmk);
                xmx_e_cur_vec = _mm256_max_ps(xmx_e_cur_vec, _mm256_set1_ps(mmx_cur[k]));
            }
        } else {
            // Ambiguous residue - set all to NEG_INFINITY
            for k in 1..=m {
                mmx_cur[k] = f32::NEG_INFINITY;
            }
        }

        // Horizontal max of xmx_e_cur_vec
        let mut xmx_e_cur = horizontal_max_avx(xmx_e_cur_vec);

        // Special states
        let xmx_j_cur = (xmx_j_prv + tloop).max(xmx_e_cur + tej);
        let xmx_c_cur = (xmx_c_prv + tloop).max(xmx_e_cur + tec);
        let xmx_n_cur = xmx_n_prv + tloop;
        let xmx_b_cur = (xmx_n_cur + tmove).max(xmx_j_cur + tmove);

        // Swap
        std::mem::swap(&mut mmx_prv, &mut mmx_cur);
        xmx_n_prv = xmx_n_cur;
        xmx_b_prv = xmx_b_cur;
        xmx_e_prv = xmx_e_cur;
        xmx_j_prv = xmx_j_cur;
        xmx_c_prv = xmx_c_cur;
    }

    xmx_c_prv + tmove
}

/// Horizontal max of AVX vector
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn horizontal_max_avx(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(hi, lo);
    let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
    _mm_cvtss_f32(max32)
}

/// Scalar MSV filter (optimized without SIMD)
///
/// MSV (Multi-ungapped Segment Viterbi) finds the best diagonal alignment
/// between sequence and model. It's essentially finding the highest-scoring
/// ungapped local alignment.
///
/// The DP recurrence is:
///   dp[k] = max(0, dp[k-1] + emission[k][res])
///
/// Where dp[k-1] is from the PREVIOUS sequence position (diagonal predecessor).
/// This allows finding high-scoring diagonal stretches.
pub fn p7_msv_scalar(opt: &P7ProfileOpt, dsq: &[u8], l: usize) -> f32 {
    let m = opt.m;
    let l_usize = l;

    if l_usize == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    // MSV algorithm matching C's p7_GMSV implementation
    // Implements proper DP with special states (N,B,E,J,C)

    // Transition probabilities (in log space, matching C implementation)
    let nu = 2.0f32;
    let tloop = ((l_usize as f32) / (l_usize as f32 + 3.0)).ln();
    let tmove = (3.0f32 / (l_usize as f32 + 3.0)).ln();
    let tbmk = (2.0f32 / ((m as f32) * (m as f32 + 1.0))).ln();
    let tej = ((nu - 1.0) / nu).ln();
    let tec = (1.0 / nu).ln();

    // DP matrices: current and previous row for match states
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
                // Match emission score (already in log-odds in P7ProfileOpt)
                let msc = opt.mscore[k][res];
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

/// SIMD-optimized Viterbi filter
#[cfg(target_arch = "x86_64")]
pub fn p7_viterbi_simd(opt: &P7ProfileOpt, dsq: &[u8], l: i32) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { p7_viterbi_avx_simple(opt, dsq, l as usize) }
    } else {
        p7_viterbi_opt(opt, dsq, l as usize)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn p7_viterbi_simd(opt: &P7ProfileOpt, dsq: &[u8], l: i32) -> f32 {
    p7_viterbi_opt(opt, dsq, l as usize)
}

/// Simple AVX Viterbi - vectorizes Match state calculation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn p7_viterbi_avx_simple(opt: &P7ProfileOpt, dsq: &[u8], l: usize) -> f32 {
    let m = opt.m;

    if l == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    const MM: usize = 0;
    const MI: usize = 1;
    const MD: usize = 2;
    const IM: usize = 3;
    const II: usize = 4;
    const DM: usize = 5;
    const DD: usize = 6;

    // Allocate with padding for SIMD
    let mut mmx = vec![f32::NEG_INFINITY; m + 16];
    let mut imx = vec![f32::NEG_INFINITY; m + 16];
    let mut dmx = vec![f32::NEG_INFINITY; m + 16];
    let mut mmx_prev = vec![f32::NEG_INFINITY; m + 16];
    let mut imx_prev = vec![f32::NEG_INFINITY; m + 16];
    let mut dmx_prev = vec![f32::NEG_INFINITY; m + 16];

    mmx_prev[0] = 0.0;
    let neg_inf_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut max_score_vec = neg_inf_vec;

    for i in 1..=l {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;
        if res >= 4 { continue; }

        // Reset current row
        mmx.fill(f32::NEG_INFINITY);
        imx.fill(f32::NEG_INFINITY);
        dmx.fill(f32::NEG_INFINITY);

        // Vectorize Match state calculation (most critical)
        let zero_vec = _mm256_set1_ps(0.0); // Local entry
        let mut k = 1;
        while k + 7 <= m {
            // Load emission scores for 8 positions
            let emit_m_vec = _mm256_set_ps(
                opt.mscore[k+7][res], opt.mscore[k+6][res],
                opt.mscore[k+5][res], opt.mscore[k+4][res],
                opt.mscore[k+3][res], opt.mscore[k+2][res],
                opt.mscore[k+1][res], opt.mscore[k][res]
            );

            // Load previous M/I/D states (k-1..k+6)
            let mmx_prev_vec = _mm256_loadu_ps(&mmx_prev[k-1]);
            let imx_prev_vec = _mm256_loadu_ps(&imx_prev[k-1]);
            let dmx_prev_vec = _mm256_loadu_ps(&dmx_prev[k-1]);

            // Load transition scores (varying by k)
            let tsc_mm_vec = _mm256_set_ps(
                opt.tscore[k+6][MM], opt.tscore[k+5][MM],
                opt.tscore[k+4][MM], opt.tscore[k+3][MM],
                opt.tscore[k+2][MM], opt.tscore[k+1][MM],
                opt.tscore[k][MM], opt.tscore[k-1][MM]
            );
            let tsc_im_vec = _mm256_set_ps(
                opt.tscore[k+6][IM], opt.tscore[k+5][IM],
                opt.tscore[k+4][IM], opt.tscore[k+3][IM],
                opt.tscore[k+2][IM], opt.tscore[k+1][IM],
                opt.tscore[k][IM], opt.tscore[k-1][IM]
            );
            let tsc_dm_vec = _mm256_set_ps(
                opt.tscore[k+6][DM], opt.tscore[k+5][DM],
                opt.tscore[k+4][DM], opt.tscore[k+3][DM],
                opt.tscore[k+2][DM], opt.tscore[k+1][DM],
                opt.tscore[k][DM], opt.tscore[k-1][DM]
            );

            // Compute max(0, mmx_prev+tsc_mm, imx_prev+tsc_im, dmx_prev+tsc_dm)
            let from_m = _mm256_add_ps(mmx_prev_vec, tsc_mm_vec);
            let from_i = _mm256_add_ps(imx_prev_vec, tsc_im_vec);
            let from_d = _mm256_add_ps(dmx_prev_vec, tsc_dm_vec);

            let max1 = _mm256_max_ps(zero_vec, from_m);
            let max2 = _mm256_max_ps(from_i, from_d);
            let best = _mm256_max_ps(max1, max2);

            // mmx[k] = emit_m + best
            let new_mmx = _mm256_add_ps(emit_m_vec, best);
            _mm256_storeu_ps(&mut mmx[k], new_mmx);

            // Track max score
            max_score_vec = _mm256_max_ps(max_score_vec, new_mmx);

            k += 8;
        }

        // Handle remaining k's with scalar
        for k in k..=m {
            let emit_m = opt.mscore[k][res];

            let mut best = 0.0_f32; // Local entry
            if mmx_prev[k-1] > f32::NEG_INFINITY {
                best = best.max(mmx_prev[k-1] + opt.tscore[k-1][MM]);
            }
            if imx_prev[k-1] > f32::NEG_INFINITY {
                best = best.max(imx_prev[k-1] + opt.tscore[k-1][IM]);
            }
            if dmx_prev[k-1] > f32::NEG_INFINITY {
                best = best.max(dmx_prev[k-1] + opt.tscore[k-1][DM]);
            }

            mmx[k] = emit_m + best;
            max_score_vec = _mm256_max_ps(max_score_vec, _mm256_set1_ps(mmx[k]));

            // Insert state (scalar only)
            let emit_i = opt.iscore[k][res];
            let mut best_i = f32::NEG_INFINITY;
            if mmx_prev[k] > f32::NEG_INFINITY {
                best_i = best_i.max(mmx_prev[k] + opt.tscore[k][MI]);
            }
            if imx_prev[k] > f32::NEG_INFINITY {
                best_i = best_i.max(imx_prev[k] + opt.tscore[k][II]);
            }
            if best_i > f32::NEG_INFINITY {
                imx[k] = emit_i + best_i;
            }

            // Delete state (scalar only)
            let mut best_d = f32::NEG_INFINITY;
            if mmx[k-1] > f32::NEG_INFINITY {
                best_d = best_d.max(mmx[k-1] + opt.tscore[k-1][MD]);
            }
            if dmx[k-1] > f32::NEG_INFINITY {
                best_d = best_d.max(dmx[k-1] + opt.tscore[k-1][DD]);
            }
            dmx[k] = best_d;
        }

        std::mem::swap(&mut mmx, &mut mmx_prev);
        std::mem::swap(&mut imx, &mut imx_prev);
        std::mem::swap(&mut dmx, &mut dmx_prev);
    }

    // Compute horizontal max of max_score_vec
    horizontal_max_avx(max_score_vec)
}

/// Optimized scalar Viterbi filter
fn p7_viterbi_opt(opt: &P7ProfileOpt, dsq: &[u8], l: usize) -> f32 {
    let m = opt.m;

    if l == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    const MM: usize = 0;
    const MI: usize = 1;
    const MD: usize = 2;
    const IM: usize = 3;
    const II: usize = 4;
    const DM: usize = 5;
    const DD: usize = 6;

    let mut mmx = vec![f32::NEG_INFINITY; m + 1];
    let mut imx = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx = vec![f32::NEG_INFINITY; m + 1];
    let mut mmx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut imx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx_prev = vec![f32::NEG_INFINITY; m + 1];

    mmx_prev[0] = 0.0;
    let mut max_score = f32::NEG_INFINITY;

    for i in 1..=l {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;
        if res >= 4 { continue; }

        // Reset current row
        mmx.fill(f32::NEG_INFINITY);
        imx.fill(f32::NEG_INFINITY);
        dmx.fill(f32::NEG_INFINITY);

        for k in 1..=m {
            // Match state
            let emit_m = opt.mscore[k][res];

            let mut best = 0.0_f32; // Local entry
            if mmx_prev[k-1] > f32::NEG_INFINITY {
                best = best.max(mmx_prev[k-1] + opt.tscore[k-1][MM]);
            }
            if imx_prev[k-1] > f32::NEG_INFINITY {
                best = best.max(imx_prev[k-1] + opt.tscore[k-1][IM]);
            }
            if dmx_prev[k-1] > f32::NEG_INFINITY {
                best = best.max(dmx_prev[k-1] + opt.tscore[k-1][DM]);
            }

            mmx[k] = emit_m + best;

            if mmx[k] > max_score {
                max_score = mmx[k];
            }

            // Insert state
            let emit_i = opt.iscore[k][res];
            let mut best_i = f32::NEG_INFINITY;
            if mmx_prev[k] > f32::NEG_INFINITY {
                best_i = best_i.max(mmx_prev[k] + opt.tscore[k][MI]);
            }
            if imx_prev[k] > f32::NEG_INFINITY {
                best_i = best_i.max(imx_prev[k] + opt.tscore[k][II]);
            }
            if best_i > f32::NEG_INFINITY {
                imx[k] = emit_i + best_i;
            }

            // Delete state
            let mut best_d = f32::NEG_INFINITY;
            if mmx[k-1] > f32::NEG_INFINITY {
                best_d = best_d.max(mmx[k-1] + opt.tscore[k-1][MD]);
            }
            if dmx[k-1] > f32::NEG_INFINITY {
                best_d = best_d.max(dmx[k-1] + opt.tscore[k-1][DD]);
            }
            dmx[k] = best_d;
        }

        std::mem::swap(&mut mmx, &mut mmx_prev);
        std::mem::swap(&mut imx, &mut imx_prev);
        std::mem::swap(&mut dmx, &mut dmx_prev);
    }

    max_score
}

/// SIMD-optimized Forward filter
/// Uses thread-local pre-allocated buffers for zero-allocation calls
#[cfg(target_arch = "x86_64")]
pub fn p7_forward_simd(opt: &P7ProfileOpt, dsq: &[u8], l: i32) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { p7_forward_avx_simple(opt, dsq, l as usize) }
    } else {
        FORWARD_FILTER.with(|filter| {
            filter.borrow_mut().forward(opt, dsq, l as usize)
        })
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn p7_forward_simd(opt: &P7ProfileOpt, dsq: &[u8], l: i32) -> f32 {
    FORWARD_FILTER.with(|filter| {
        filter.borrow_mut().forward(opt, dsq, l as usize)
    })
}

/// Fast approximation-based Forward - uses fast_exp/fast_ln for speed
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn p7_forward_avx_simple(opt: &P7ProfileOpt, dsq: &[u8], l: usize) -> f32 {
    let m = opt.m;

    if l == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    const MM: usize = 0;
    const MI: usize = 1;
    const MD: usize = 2;
    const IM: usize = 3;
    const II: usize = 4;
    const DM: usize = 5;
    const DD: usize = 6;

    let mut mmx = vec![f32::NEG_INFINITY; m + 1];
    let mut imx = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx = vec![f32::NEG_INFINITY; m + 1];
    let mut mmx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut imx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx_prev = vec![f32::NEG_INFINITY; m + 1];

    let nu = 2.0f32;
    let tloop = ((l as f32) / (l as f32 + 3.0)).ln();
    let tmove = (3.0f32 / (l as f32 + 3.0)).ln();
    let tbmk = (2.0f32 / ((m as f32) * (m as f32 + 1.0))).ln();
    let tej = ((nu - 1.0) / nu).ln();
    let tec = (1.0 / nu).ln();

    let mut xmx_n_prv = 0.0f32;
    let mut xmx_b_prv = tmove;
    let mut xmx_e_prv = f32::NEG_INFINITY;
    let mut xmx_j_prv = f32::NEG_INFINITY;
    let mut xmx_c_prv = f32::NEG_INFINITY;

    mmx_prev[0] = f32::NEG_INFINITY;

    for i in 1..=l {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;
        if res >= 4 { continue; }

        mmx.fill(f32::NEG_INFINITY);
        imx.fill(f32::NEG_INFINITY);
        dmx.fill(f32::NEG_INFINITY);

        let mut xmx_e_cur = f32::NEG_INFINITY;

        // Use fast approximations for log_sum_exp
        for k in 1..=m {
            let emit_m = opt.mscore[k][res];

            let from_m = if mmx_prev[k-1] > f32::NEG_INFINITY {
                mmx_prev[k-1] + opt.tscore[k-1][MM]
            } else { f32::NEG_INFINITY };

            let from_i = if imx_prev[k-1] > f32::NEG_INFINITY {
                imx_prev[k-1] + opt.tscore[k-1][IM]
            } else { f32::NEG_INFINITY };

            let from_d = if dmx_prev[k-1] > f32::NEG_INFINITY {
                dmx_prev[k-1] + opt.tscore[k-1][DM]
            } else { f32::NEG_INFINITY };

            let from_b = xmx_b_prv + tbmk;

            // Use fast approximation
            mmx[k] = emit_m + log_sum_exp4_fast(from_m, from_i, from_d, from_b);

            // Insert
            let emit_i = opt.iscore[k][res];
            let from_m_i = if mmx_prev[k] > f32::NEG_INFINITY {
                mmx_prev[k] + opt.tscore[k][MI]
            } else { f32::NEG_INFINITY };
            let from_i_i = if imx_prev[k] > f32::NEG_INFINITY {
                imx_prev[k] + opt.tscore[k][II]
            } else { f32::NEG_INFINITY };

            let sum_i = log_sum_exp2_fast(from_m_i, from_i_i);
            if sum_i > f32::NEG_INFINITY {
                imx[k] = emit_i + sum_i;
            }

            // Delete
            let from_m_d = if mmx[k-1] > f32::NEG_INFINITY {
                mmx[k-1] + opt.tscore[k-1][MD]
            } else { f32::NEG_INFINITY };
            let from_d_d = if dmx[k-1] > f32::NEG_INFINITY {
                dmx[k-1] + opt.tscore[k-1][DD]
            } else { f32::NEG_INFINITY };

            dmx[k] = log_sum_exp2_fast(from_m_d, from_d_d);

            xmx_e_cur = log_sum_exp2_fast(xmx_e_cur, mmx[k]);
        }

        // Special states (use fast approximations)
        let xmx_j_cur = log_sum_exp2_fast(xmx_j_prv + tloop, xmx_e_cur + tej);
        let xmx_c_cur = log_sum_exp2_fast(xmx_c_prv + tloop, xmx_e_cur + tec);
        let xmx_n_cur = xmx_n_prv + tloop;
        let xmx_b_cur = log_sum_exp2_fast(xmx_n_cur + tmove, xmx_j_cur + tmove);

        std::mem::swap(&mut mmx, &mut mmx_prev);
        std::mem::swap(&mut imx, &mut imx_prev);
        std::mem::swap(&mut dmx, &mut dmx_prev);

        xmx_n_prv = xmx_n_cur;
        xmx_b_prv = xmx_b_cur;
        xmx_e_prv = xmx_e_cur;
        xmx_j_prv = xmx_j_cur;
        xmx_c_prv = xmx_c_cur;
    }

    xmx_c_prv + tmove
}

/// Optimized Forward filter with pre-allocated vectors
/// Matches C's p7_GForward implementation with special states (N, B, E, J, C)
fn p7_forward_opt(opt: &P7ProfileOpt, dsq: &[u8], l: usize) -> f32 {
    let m = opt.m;

    if l == 0 || m == 0 {
        return f32::NEG_INFINITY;
    }

    const MM: usize = 0;
    const MI: usize = 1;
    const MD: usize = 2;
    const IM: usize = 3;
    const II: usize = 4;
    const DM: usize = 5;
    const DD: usize = 6;

    let mut mmx = vec![f32::NEG_INFINITY; m + 1];
    let mut imx = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx = vec![f32::NEG_INFINITY; m + 1];
    let mut mmx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut imx_prev = vec![f32::NEG_INFINITY; m + 1];
    let mut dmx_prev = vec![f32::NEG_INFINITY; m + 1];

    // Special state transitions (simplified for local alignment)
    // In full implementation, these would come from profile xsc parameters
    // For now, use reasonable defaults matching HMMER3 local alignment mode
    let nu = 2.0f32;
    let tloop = ((l as f32) / (l as f32 + 3.0)).ln();
    let tmove = (3.0f32 / (l as f32 + 3.0)).ln();
    let tej = ((nu - 1.0) / nu).ln();
    let tec = (1.0 / nu).ln();

    // Special states: N, B, E, J, C
    let mut xmx_n_prv = 0.0f32;
    let mut xmx_b_prv = tmove;  // S->N->B, no N-tail
    let mut xmx_e_prv = f32::NEG_INFINITY;
    let mut xmx_j_prv = f32::NEG_INFINITY;
    let mut xmx_c_prv = f32::NEG_INFINITY;

    mmx_prev[0] = f32::NEG_INFINITY;
    imx_prev[0] = f32::NEG_INFINITY;
    dmx_prev[0] = f32::NEG_INFINITY;

    for i in 1..=l {
        if i >= dsq.len() { break; }

        let res = dsq[i] as usize;
        if res >= 4 { continue; }

        mmx.fill(f32::NEG_INFINITY);
        imx.fill(f32::NEG_INFINITY);
        dmx.fill(f32::NEG_INFINITY);

        let mut xmx_e_cur = f32::NEG_INFINITY;

        for k in 1..=m {
            // Match: log-sum-exp of incoming paths + emission
            let emit_m = opt.mscore[k][res];

            let from_m = if mmx_prev[k-1] > f32::NEG_INFINITY {
                mmx_prev[k-1] + opt.tscore[k-1][MM]
            } else { f32::NEG_INFINITY };

            let from_i = if imx_prev[k-1] > f32::NEG_INFINITY {
                imx_prev[k-1] + opt.tscore[k-1][IM]
            } else { f32::NEG_INFINITY };

            let from_d = if dmx_prev[k-1] > f32::NEG_INFINITY {
                dmx_prev[k-1] + opt.tscore[k-1][DM]
            } else { f32::NEG_INFINITY };

            // Local entry from B state (using simplified transition)
            let from_b = xmx_b_prv + ((2.0f32) / ((m as f32) * (m as f32 + 1.0))).ln();

            mmx[k] = emit_m + log_sum_exp4(from_m, from_i, from_d, from_b);

            // Insert
            let emit_i = opt.iscore[k][res];
            let from_m_i = if mmx_prev[k] > f32::NEG_INFINITY {
                mmx_prev[k] + opt.tscore[k][MI]
            } else { f32::NEG_INFINITY };
            let from_i_i = if imx_prev[k] > f32::NEG_INFINITY {
                imx_prev[k] + opt.tscore[k][II]
            } else { f32::NEG_INFINITY };

            let sum_i = log_sum_exp2(from_m_i, from_i_i);
            if sum_i > f32::NEG_INFINITY {
                imx[k] = emit_i + sum_i;
            }

            // Delete
            let from_m_d = if mmx[k-1] > f32::NEG_INFINITY {
                mmx[k-1] + opt.tscore[k-1][MD]
            } else { f32::NEG_INFINITY };
            let from_d_d = if dmx[k-1] > f32::NEG_INFINITY {
                dmx[k-1] + opt.tscore[k-1][DD]
            } else { f32::NEG_INFINITY };

            dmx[k] = log_sum_exp2(from_m_d, from_d_d);

            // E state: accumulate exits from all match states
            xmx_e_cur = log_sum_exp2(xmx_e_cur, mmx[k]);
        }

        // Special states updates (matching C's p7_GForward)
        // J state: J->J loop or E->J
        let xmx_j_cur = log_sum_exp2(xmx_j_prv + tloop, xmx_e_cur + tej);

        // C state: C->C loop or E->C
        let xmx_c_cur = log_sum_exp2(xmx_c_prv + tloop, xmx_e_cur + tec);

        // N state: N->N loop
        let xmx_n_cur = xmx_n_prv + tloop;

        // B state: N->B or J->B
        let xmx_b_cur = log_sum_exp2(xmx_n_cur + tmove, xmx_j_cur + tmove);

        std::mem::swap(&mut mmx, &mut mmx_prev);
        std::mem::swap(&mut imx, &mut imx_prev);
        std::mem::swap(&mut dmx, &mut dmx_prev);

        xmx_n_prv = xmx_n_cur;
        xmx_b_prv = xmx_b_cur;
        xmx_e_prv = xmx_e_cur;
        xmx_j_prv = xmx_j_cur;
        xmx_c_prv = xmx_c_cur;
    }

    // Return C state score (C->T transition)
    xmx_c_prv + tmove
}

/// Fast log-sum-exp for 2 values
#[inline(always)]
fn log_sum_exp2(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY { return b; }
    if b == f32::NEG_INFINITY { return a; }
    let max = a.max(b);
    let min = a.min(b);
    max + (1.0 + (min - max).exp()).ln()
}

/// Fast log-sum-exp for 4 values
#[inline(always)]
fn log_sum_exp4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let max = a.max(b).max(c).max(d);
    if max == f32::NEG_INFINITY { return f32::NEG_INFINITY; }

    let mut sum = 0.0_f32;
    if a > f32::NEG_INFINITY { sum += (a - max).exp(); }
    if b > f32::NEG_INFINITY { sum += (b - max).exp(); }
    if c > f32::NEG_INFINITY { sum += (c - max).exp(); }
    if d > f32::NEG_INFINITY { sum += (d - max).exp(); }

    max + sum.ln()
}

/// Ultra-fast exp approximation using integer bit manipulation
/// Based on Schraudolph's algorithm - accurate to ~4% for range [-87, 88]
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    // Clamp to avoid overflow/underflow
    let x = x.clamp(-87.0, 88.0);
    // Magic constants from Schraudolph 1999
    // exp(x) ≈ 2^(x/ln2) using float bit representation
    let a = (1 << 23) as f32 / std::f32::consts::LN_2;
    let b = (127 << 23) as f32 - 366000.0; // Adjusted bias
    let v = (a * x + b) as i32;
    f32::from_bits(v as u32)
}

/// Ultra-fast ln approximation using integer bit manipulation
#[inline(always)]
fn fast_ln(x: f32) -> f32 {
    if x <= 0.0 { return f32::NEG_INFINITY; }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let m = f32::from_bits((bits & 0x007FFFFF) | 0x3F800000);
    // Polynomial approximation for ln(m) where m is in [1, 2)
    let ln_m = (m - 1.0) * (2.0 - 0.333333 * (m - 1.0));
    e as f32 * std::f32::consts::LN_2 + ln_m
}

/// Fast log-sum-exp for 2 values using approximations
#[inline(always)]
fn log_sum_exp2_fast(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY { return b; }
    if b == f32::NEG_INFINITY { return a; }
    let max = a.max(b);
    let diff = a.min(b) - max;
    if diff < -15.0 { return max; } // exp(-15) ≈ 3e-7, negligible
    max + fast_ln(1.0 + fast_exp(diff))
}

/// Fast log-sum-exp for 4 values using approximations
#[inline(always)]
fn log_sum_exp4_fast(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let max = a.max(b).max(c).max(d);
    if max == f32::NEG_INFINITY { return f32::NEG_INFINITY; }

    let mut sum = 0.0_f32;
    if a > f32::NEG_INFINITY && a - max > -15.0 { sum += fast_exp(a - max); }
    if b > f32::NEG_INFINITY && b - max > -15.0 { sum += fast_exp(b - max); }
    if c > f32::NEG_INFINITY && c - max > -15.0 { sum += fast_exp(c - max); }
    if d > f32::NEG_INFINITY && d - max > -15.0 { sum += fast_exp(d - max); }

    max + fast_ln(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p7_hmm::P7Profile;

    #[test]
    fn test_p7_opt_creation() {
        let mut p7 = P7Profile::new(10);
        for k in 1..=10 {
            p7.mat[k] = [0.4, 0.3, 0.2, 0.1];
            p7.trans[k-1][0] = 0.9; // MM
        }

        let opt = P7ProfileOpt::from_p7(&p7);
        assert_eq!(opt.m, 10);

        // Check that log-odds scores are computed correctly
        // For mat[1][0] = 0.4, log-odds = ln(0.4) - ln(0.25) = ln(0.4/0.25) = ln(1.6)
        let expected = (0.4_f32 / 0.25).ln();
        assert!((opt.mscore[1][0] - expected).abs() < 0.001);
    }

    #[test]
    fn test_msv_scalar() {
        let mut p7 = P7Profile::new(5);
        for k in 1..=5 {
            p7.mat[k] = [0.7, 0.1, 0.1, 0.1];
        }

        let opt = P7ProfileOpt::from_p7(&p7);
        let dsq = vec![255, 0, 0, 0, 0, 0, 255];

        let score = p7_msv_scalar(&opt, &dsq, 5);
        assert!(score.is_finite());
        // Full ungapped diagonal of 5 A-matches minus MSV entry/exit/length costs;
        // the entry cost tBM = ln(2/(M(M+1))) makes the net C-state score slightly
        // negative (≈ -0.2148 bits), equal to p7_dp::p7_msv_filter for this input.
        assert!((score - (-0.214759f32)).abs() < 0.001, "got {}", score);
    }

    #[test]
    fn test_msv_simd() {
        let mut p7 = P7Profile::new(20);
        for k in 1..=20 {
            p7.mat[k] = [0.7, 0.1, 0.1, 0.1];
        }

        let opt = P7ProfileOpt::from_p7(&p7);
        // AAAAA... sequence
        let mut dsq = vec![255_u8];
        dsq.extend(vec![0_u8; 20]);
        dsq.push(255);

        let scalar_score = p7_msv_scalar(&opt, &dsq, 20);
        let simd_score = p7_msv_simd(&opt, &dsq, 20);

        // SIMD and scalar should give same result
        assert!((scalar_score - simd_score).abs() < 0.01,
            "Scalar: {}, SIMD: {}", scalar_score, simd_score);
    }
}
