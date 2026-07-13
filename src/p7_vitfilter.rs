//! Faithful port of HMMER's `p7_ViterbiFilter` (impl_sse/vitfilter.c) and its
//! optimized-profile conversion `vf_conversion` (impl_sse/p7_oprofile.c), for
//! Infernal's p7 filter calibration (`p7_ViterbiMu`, evalues.c:307).
//!
//! This is a natural-order (unstriped) scalar transcription that computes the
//! byte-identical final `xC`/score as the striped SSE version:
//!   - E is reached only from M cells, so `xE = max_k M(i,k)` is layout-free.
//!   - The Delete lane is computed with a full serial left→right DD sweep
//!     (`D(i,k) = max(M(i,k-1)+TMD, D(i,k-1)+TDD)`), which is exactly the value
//!     the SSE "lazy-F" loop converges to (it iterates DD passes until no cell
//!     improves), so the two agree bit-for-bit.
//!   - All arithmetic is 16-bit saturating (`i16` with `saturating_add`), with
//!     -infinity represented as -32768, matching `_mm_adds_epi16`.
//!
//! Emission/transition scores are quantized with `wordify` exactly as
//! `vf_conversion`. The RNA background is uniform (`bg->f[x] = 0.25`, K=4), so
//! the degenerate-residue expected score reduces to the simple mean over the
//! canonical residues in each IUPAC code (identical to `esl_abc_FExpectScVec`
//! under a uniform background) — matching the MSV/Forward filters in this crate.

use crate::p7_hmm::P7Profile;

const KP: usize = 18; // RNA extended alphabet (A,C,G,U,gap,degens,*,~)
const K_CANON: usize = 4;
const NEGINF: i16 = -32768;
const LOG2: f64 = std::f64::consts::LN_2; // eslCONST_LOG2

/// RNA IUPAC degeneracy → canonical residues each code expands to (see
/// `cm_pipeline::degen_set`). Empty = gap/nonresidue → -inf.
fn degen_set(code: usize) -> &'static [usize] {
    match code {
        0 => &[0], 1 => &[1], 2 => &[2], 3 => &[3],
        5 => &[0, 2],        // R = A|G
        6 => &[1, 3],        // Y = C|U
        7 => &[0, 1],        // M = A|C
        8 => &[2, 3],        // K = G|U
        9 => &[1, 2],        // S = C|G
        10 => &[0, 3],       // W = A|U
        11 => &[0, 1, 3],    // H = A|C|U
        12 => &[1, 2, 3],    // B = C|G|U
        13 => &[0, 1, 2],    // V = A|C|G
        14 => &[0, 2, 3],    // D = A|G|U
        15 => &[0, 1, 2, 3], // N = any
        _ => &[],
    }
}

/// C: impl_sse/p7_oprofile.c::wordify() — quantize a nat score to a scaled 16-bit
/// integer, saturating at [-32768, 32767].
/// ```c
/// sc = roundf(om->scale_w * sc);
/// if      (sc >=  32767.0) return  32767;
/// else if (sc <= -32768.0) return -32768;
/// else return (int16_t) sc;
/// ```
#[inline]
fn wordify(scale_w: f32, sc: f32) -> i16 {
    if sc == f32::NEG_INFINITY {
        return NEGINF;
    }
    let v = (scale_w * sc).round();
    if v >= 32767.0 {
        32767
    } else if v <= -32768.0 {
        NEGINF
    } else {
        v as i16
    }
}

/// C: `_mm_adds_epi16` — signed 16-bit saturating add.
#[inline]
fn adds(a: i16, b: i16) -> i16 {
    a.saturating_add(b)
}

/// Configured Viterbi filter profile (LOCAL multihit). Length-independent parts;
/// the N/C/J length model (`xw_move`) is applied per sequence length in
/// [`vit_score`], mirroring `p7_oprofile_ReconfigLength`.
pub struct VitFilter {
    pub m: usize,
    pub scale_w: f32,
    pub base_w: i16,
    /// rwv[k][x] = wordify(MSC[k][x]); k=1..=M.
    rwv: Vec<[i16; KP]>,
    /// Into-M transitions from node k-1: tmm=M->M, tim=I->M, tdm=D->M; k=1..=M-1.
    tmm: Vec<i16>,
    tim: Vec<i16>,
    tdm: Vec<i16>,
    /// Local entry B->M_k = wordify(log(occ[k]/Z)) capped at 0; k=1..=M.
    vbm: Vec<i16>,
    /// Insert transitions at node k: tmi=M->I, tii=I->I (II capped at -1); k=1..=M-1.
    tmi: Vec<i16>,
    tii: Vec<i16>,
    /// Delete-out transitions of node k: tmd=M_k->D_{k+1}, tdd=D_k->D_{k+1}; k=1..=M-1.
    tmd: Vec<i16>,
    tdd: Vec<i16>,
    /// E->C (MOVE) and E->J (LOOP) both = wordify(-log2).
    xw_e_move: i16,
    xw_e_loop: i16,
}

/// Build the Viterbi filter from a p7 filter HMM. Faithful to
/// `p7_ProfileConfig(LOCAL)` + `vf_conversion` (impl_sse/p7_oprofile.c).
/// `p7.trans` layout is [MM,MI,MD,IM,II,DM,DD].
pub fn build_vit_filter(p7: &P7Profile) -> VitFilter {
    let m = p7.m as usize;

    // vf_conversion:518-519.
    let scale_w = (500.0_f64 / LOG2) as f32;
    let base_w: i16 = 12000;

    // Striped match scores: rwv[k][x] = wordify(MSC[k][x]) where
    // MSC[k][x] = log(mat[k][x]/0.25) (modelconfig.c:141-151); degenerates via
    // esl_abc_FExpectScVec = simple mean under uniform bg.
    let mut rwv = vec![[NEGINF; KP]; m + 1];
    for k in 1..=m {
        let mut sc = [f32::NEG_INFINITY; KP];
        for x in 0..K_CANON {
            sc[x] = ((p7.mat[k][x] as f64) / 0.25).ln() as f32;
        }
        for code in K_CANON..KP {
            let set = degen_set(code);
            if set.is_empty() {
                continue;
            }
            let mut s = 0.0f32;
            for &x in set {
                s += sc[x];
            }
            sc[code] = s / set.len() as f32;
        }
        for x in 0..KP {
            rwv[k][x] = wordify(scale_w, sc[x]);
        }
    }

    // Transition costs (vf_conversion:527-560). The into-M transitions (MM/IM/DM)
    // are stored off-by-one (kb=k-1). In natural-order arrays: tmm[j] holds the
    // M_j->M_{j+1} transition = log(t[j][MM]); i.e. tmm[k-1] feeds M(i,k). The
    // `maxval` cap forbids a zero-cost II transition (and clamps others to <=0).
    // p7.trans indices: MM=0, MI=1, MD=2, IM=3, II=4, DM=5, DD=6.
    let mut tmm = vec![NEGINF; m + 1];
    let mut tim = vec![NEGINF; m + 1];
    let mut tdm = vec![NEGINF; m + 1];
    let mut tmi = vec![NEGINF; m + 1];
    let mut tii = vec![NEGINF; m + 1];
    let mut tmd = vec![NEGINF; m + 1];
    let mut tdd = vec![NEGINF; m + 1];
    for j in 1..m {
        tmm[j] = wordify(scale_w, (p7.trans[j][0] as f64).ln() as f32).min(0); // MM
        tim[j] = wordify(scale_w, (p7.trans[j][3] as f64).ln() as f32).min(0); // IM
        tdm[j] = wordify(scale_w, (p7.trans[j][5] as f64).ln() as f32).min(0); // DM
        tmi[j] = wordify(scale_w, (p7.trans[j][1] as f64).ln() as f32).min(0); // MI
        tii[j] = wordify(scale_w, (p7.trans[j][4] as f64).ln() as f32).min(-1); // II -> cap -1
        tmd[j] = wordify(scale_w, (p7.trans[j][2] as f64).ln() as f32).min(0); // MD
        tdd[j] = wordify(scale_w, (p7.trans[j][6] as f64).ln() as f32).min(0); // DD
    }

    // Local entry occupancy (p7_hmm_CalculateOccupancy) then B->M_k = occ[k]/Z
    // (modelconfig.c:90-97). Same computation as the Forward filter.
    let mut occ = vec![0.0f32; m + 1];
    if m >= 1 {
        occ[1] = p7.trans[0][1] + p7.trans[0][0]; // MI + MM
    }
    for k in 2..=m {
        occ[k] = occ[k - 1] * (p7.trans[k - 1][0] + p7.trans[k - 1][1])
            + (1.0 - occ[k - 1]) * p7.trans[k - 1][5];
    }
    let mut z = 0.0f32;
    for k in 1..=m {
        z += occ[k] * (m - k + 1) as f32;
    }
    let mut vbm = vec![NEGINF; m + 1];
    for k in 1..=m {
        let logv = ((occ[k] / z) as f64).ln() as f32;
        vbm[k] = wordify(scale_w, logv).min(0);
    }

    // Specials: E moves. VF hardcodes NN/CC/JJ = 0 (the -3nat approximation).
    let xw_e_move = wordify(scale_w, -(LOG2 as f32));
    let xw_e_loop = wordify(scale_w, -(LOG2 as f32));

    VitFilter {
        m,
        scale_w,
        base_w,
        rwv,
        tmm,
        tim,
        tdm,
        vbm,
        tmi,
        tii,
        tmd,
        tdd,
        xw_e_move,
        xw_e_loop,
    }
}

/// Whole-sequence Viterbi filter score in bits, faithful to `p7_ViterbiFilter`
/// (impl_sse/vitfilter.c:83). Returns `None` on the eslERANGE overflow
/// (`xE >= 32767`); the calibration caller then substitutes
/// `maxsc = (32767 - base_w)/scale_w`.
///
/// `dsq` is 1-indexed with sentinels. The length model is (multihit) LOCAL:
/// `pmove = (2+nj)/(L+2+nj)` with `nj=1` ⇒ `3/(L+3)`; N/C/J LOOP costs are 0.
pub fn vit_score(f: &VitFilter, dsq: &[u8], l: usize) -> Option<f32> {
    let m = f.m;

    // p7_oprofile_ReconfigLength: xw[N/C/J][MOVE] = wordify(log(pmove)), pmove =
    // (2+nj)/(L+2+nj), nj=1 (multihit) => 3/(L+3). LOOP costs stay 0.
    let pmove = 3.0_f32 / (l as f32 + 3.0);
    let xw_move = wordify(f.scale_w, pmove.ln());

    // DP rows (k=0..=M). -inf everywhere at init (vitfilter.c:194-198).
    let mut mp = vec![NEGINF; m + 1];
    let mut ip = vec![NEGINF; m + 1];
    let mut dp = vec![NEGINF; m + 1];
    let mut mc = vec![NEGINF; m + 1];
    let mut ic = vec![NEGINF; m + 1];
    let mut dc = vec![NEGINF; m + 1];

    let mut xn: i16 = f.base_w;
    let mut xb: i16 = adds(xn, xw_move);
    let mut xj: i16 = NEGINF;
    let mut xc: i16 = NEGINF;

    for i in 1..=l {
        let x = dsq[i] as usize;
        mc[0] = NEGINF;
        ic[0] = NEGINF;
        dc[0] = NEGINF;

        // M states: M(i,k) = max(xB+BM_k, M(i-1,k-1)+MM, I(i-1,k-1)+IM,
        //                         D(i-1,k-1)+DM) + emission. (vitfilter.c:213-234)
        for k in 1..=m {
            let sc = adds(xb, f.vbm[k])
                .max(adds(mp[k - 1], f.tmm[k - 1]))
                .max(adds(ip[k - 1], f.tim[k - 1]))
                .max(adds(dp[k - 1], f.tdm[k - 1]));
            mc[k] = adds(sc, f.rwv[k][x]);
        }

        // I states: I(i,k) = max(M(i-1,k)+MI, I(i-1,k)+II). (vitfilter.c:245-246)
        for k in 1..=m {
            ic[k] = adds(mp[k], f.tmi[k]).max(adds(ip[k], f.tii[k]));
        }

        // xE = max over M cells (E is reached only from M). (vitfilter.c:175)
        let mut xe: i16 = NEGINF;
        for &v in &mc[1..=m] {
            if v > xe {
                xe = v;
            }
        }

        // D states, full serial DD sweep (the value the SSE lazy-F converges to).
        // D(i,k) = max(M(i,k-1)+MD_{k-1}, D(i,k-1)+DD_{k-1}). (vitfilter.c:227-234 + lazy-F)
        for k in 1..=m {
            dc[k] = adds(mc[k - 1], f.tmd[k - 1]).max(adds(dc[k - 1], f.tdd[k - 1]));
        }

        // Overflow detection (vitfilter.c:176).
        if xe >= 32767 {
            return None; // eslERANGE
        }

        // Specials (vitfilter.c:177-181). N/C/J loops are 0.
        xn = adds(xn, 0);
        xc = xc.max(adds(xe, f.xw_e_move));
        xj = xj.max(adds(xe, f.xw_e_loop));
        xb = adds(xj, xw_move).max(adds(xn, xw_move));

        std::mem::swap(&mut mp, &mut mc);
        std::mem::swap(&mut ip, &mut ic);
        std::mem::swap(&mut dp, &mut dc);
    }

    // C->T (vitfilter.c:243-252).
    if xc > NEGINF {
        let mut ret = xc as f32 + xw_move as f32 - f.base_w as f32;
        ret /= f.scale_w;
        Some(ret - 3.0)
    } else {
        Some(f32::NEG_INFINITY)
    }
}

impl VitFilter {
    /// `(32767 - base_w)/scale_w` — the score assigned on eslERANGE overflow
    /// (evalues.c::p7_ViterbiMu:308).
    pub fn cal_maxsc(&self) -> f32 {
        (32767.0 - self.base_w as f32) / self.scale_w
    }
}
