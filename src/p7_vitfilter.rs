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

    // ---- Striped SSE (Farrar) layout, built by [`stripe`] for [`vit_score_sse`]. ----
    /// Q = p7O_NQW(M) = ESL_MAX(2, (M-1)/8 + 1) — # of 8-wide i16 vectors.
    q: usize,
    /// Striped transition scores `om->twv`, one 8-i16 vector per entry, laid out
    /// exactly as `vf_conversion`: for q=0..Q the 7 interleaved vectors
    /// [BM,MM,IM,DM,MD,MI,II] at vector index `q*7+t`, then a trailing DD block of
    /// Q vectors at index `7*Q+q`. Flattened: vector `v` occupies `tw[v*8..v*8+8]`.
    tw: Vec<i16>,
    /// Striped match emission scores `om->rwv`: for residue x, vector q at
    /// `rw[(x*Q+q)*8 .. +8]`; slot z holds model position k=q+1+z*Q (or -inf if k>M).
    rw: Vec<i16>,
    /// Lazy-F DD short-circuit bound `om->ddbound_w`
    /// = max_{k=2..M-2} (TDD(k) + TDM(k+1) - TBM(k+2)). (vf_conversion:910-918)
    ddbound_w: i16,
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

    // ---- Striped SSE layout (vf_conversion) ----
    // Q = ESL_MAX(2, (M-1)/8 + 1) (impl_sse.h:25). M>=1 always here.
    let q = std::cmp::max(2, (m - 1) / 8 + 1);

    // Striped match emissions rwv[x][q][z], k = q+1 + z*Q (vf_conversion:852-858).
    // Condition k<=M else -inf. rwv_flat is indexed [k][x].
    let mut rw = vec![NEGINF; KP * q * 8];
    for x in 0..KP {
        for qi in 0..q {
            for z in 0..8 {
                let k = (qi + 1) + z * q; // model position
                rw[(x * q + qi) * 8 + z] = if k <= m { rwv[k][x] } else { NEGINF };
            }
        }
    }

    // Striped transitions twv (vf_conversion:860-888). Interleaved 7 vectors per q
    // [BM,MM,IM,DM,MD,MI,II] then a DD block of Q vectors. Into-M (BM/MM/IM/DM) use
    // base position k=q+1 with condition k<=M and read t[k-1] (rotated -1); the rest
    // (MD/MI/II/DD) use k=q+1 with condition k<M and read t[k].
    let mut tw = vec![NEGINF; (7 * q + q) * 8];
    for qi in 0..q {
        for z in 0..8 {
            let k = (qi + 1) + z * q; // model position
            let vbase = qi * 7;
            // BM,MM,IM,DM: k<=M, index [k]/[k-1]
            let (bm, mm, im, dm) = if k <= m {
                (vbm[k], tmm[k - 1], tim[k - 1], tdm[k - 1])
            } else {
                (NEGINF, NEGINF, NEGINF, NEGINF)
            };
            tw[(vbase + 0) * 8 + z] = bm;
            tw[(vbase + 1) * 8 + z] = mm;
            tw[(vbase + 2) * 8 + z] = im;
            tw[(vbase + 3) * 8 + z] = dm;
            // MD,MI,II: k<M, index [k]
            let (md, mi, ii) = if k < m {
                (tmd[k], tmi[k], tii[k])
            } else {
                (NEGINF, NEGINF, NEGINF)
            };
            tw[(vbase + 4) * 8 + z] = md;
            tw[(vbase + 5) * 8 + z] = mi;
            tw[(vbase + 6) * 8 + z] = ii;
            // DD block (k<M, index [k]).
            tw[(7 * q + qi) * 8 + z] = if k < m { tdd[k] } else { NEGINF };
        }
    }

    // Lazy-F DD bound (vf_conversion:910-918): int16 running max, updated per k.
    // ddtmp = TDD(k) + TDM(k+1) - TBM(k+2); k = 2..M-2.
    let mut ddbound_w: i16 = NEGINF;
    let mut k = 2;
    while k < m.saturating_sub(1) {
        let ddtmp = tdd[k] as i32 + tdm[k + 1] as i32 - vbm[k + 2] as i32;
        ddbound_w = std::cmp::max(ddbound_w as i32, ddtmp) as i16;
        k += 1;
    }

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
        q,
        tw,
        rw,
        ddbound_w,
    }
}

/// Whole-sequence Viterbi filter score in bits, faithful to `p7_ViterbiFilter`
/// (impl_sse/vitfilter.c:83). Returns `None` on the eslERANGE overflow
/// (`xE >= 32767`); the calibration caller then substitutes
/// `maxsc = (32767 - base_w)/scale_w`.
///
/// `dsq` is 1-indexed with sentinels. The length model is (multihit) LOCAL:
/// `pmove = (2+nj)/(L+2+nj)` with `nj=1` ⇒ `3/(L+3)`; N/C/J LOOP costs are 0.
///
/// Runtime dispatcher: uses the striped-SSE kernel [`vit_score_sse`] on x86_64
/// (byte-identical output, verified by `sse_matches_scalar`), else the scalar
/// oracle [`vit_score_scalar`].
#[inline]
pub fn vit_score(f: &VitFilter, dsq: &[u8], l: usize) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: guarded by the sse2 feature check just above.
            return unsafe { vit_score_sse(f, dsq, l) };
        }
    }
    vit_score_scalar(f, dsq, l)
}

/// Natural-order scalar oracle. See the module header for why its full serial DD
/// sweep is bit-identical to the striped SSE "lazy-F" convergence.
pub fn vit_score_scalar(f: &VitFilter, dsq: &[u8], l: usize) -> Option<f32> {
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

/// Horizontal max of 8 i16 lanes — faithful `esl_sse_hmax_epi16` (esl_sse.h:75).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hmax_epi16(a: core::arch::x86_64::__m128i) -> i16 {
    use core::arch::x86_64::*;
    // _MM_SHUFFLE(1,0,3,2) == 0b01_00_11_10.
    let a = _mm_max_epi16(a, _mm_shuffle_epi32::<0b01_00_11_10>(a));
    let a = _mm_max_epi16(a, _mm_shufflelo_epi16::<0b01_00_11_10>(a));
    let a = _mm_max_epi16(a, _mm_srli_epi32::<16>(a));
    _mm_cvtsi128_si32(a) as i16
}

/// Striped SIMD (Farrar) i16 Viterbi filter — faithful port of `p7_ViterbiFilter`
/// (impl_sse/vitfilter.c:82). Byte-identical to [`vit_score_scalar`]; integer
/// saturating max/add is associative so the striped/interleaved evaluation order
/// yields the same `xE`/`xC` and thus the same final score. The lazy-F DD loop
/// converges to the same D values the scalar's full serial DD sweep computes.
///
/// SAFETY: caller must ensure the `sse2` target feature is available (guaranteed
/// on x86_64 baseline; checked in [`vit_score`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vit_score_sse(f: &VitFilter, dsq: &[u8], l: usize) -> Option<f32> {
    use core::arch::x86_64::*;
    let q = f.q;

    // p7_oprofile_ReconfigLength (see vit_score_scalar). N/C/J LOOP costs stay 0.
    let pmove = 3.0_f32 / (l as f32 + 3.0);
    let xw_move = wordify(f.scale_w, pmove.ln());

    // "-infinity" and the OR-mask (14 zero bytes + one -32768 word in the low lane)
    // used to refill the lane shifted in by _mm_slli_si128 (vitfilter.c:108-109).
    let neg_inf = _mm_set1_epi16(NEGINF);
    let neginfv = _mm_srli_si128::<14>(neg_inf);

    // Load helpers into the flat striped arrays (unaligned; vector v -> [v*8..]).
    #[inline(always)]
    unsafe fn ld(base: *const i16, v: usize) -> core::arch::x86_64::__m128i {
        core::arch::x86_64::_mm_loadu_si128(base.add(v * 8) as *const core::arch::x86_64::__m128i)
    }
    let twp = f.tw.as_ptr();
    let rwp = f.rw.as_ptr();

    // DP rows, one 8-wide vector per q, all -inf (vitfilter.c:113-114).
    let mut mmx = vec![neg_inf; q];
    let mut imx = vec![neg_inf; q];
    let mut dmx = vec![neg_inf; q];

    let mut xn: i16 = f.base_w;
    let mut xb: i16 = adds(xn, xw_move);
    let mut xj: i16 = NEGINF;
    let mut xc: i16 = NEGINF;

    let dd_base = 7 * q; // first DD vector index in `tw`

    for i in 1..=l {
        let x = dsq[i] as usize;
        let rbase = x * q; // striped residue-vector base (vectors), rwp offset = rbase*8
        let mut dcv = neg_inf;
        let mut xev = neg_inf;
        let mut dmaxv = neg_inf;
        let xbv = _mm_set1_epi16(xb);

        // Right-shift (=little-endian left) the wrapped last vector, refill lane0 with -inf.
        let mut mpv = _mm_or_si128(_mm_slli_si128::<2>(mmx[q - 1]), neginfv);
        let mut dpv = _mm_or_si128(_mm_slli_si128::<2>(dmx[q - 1]), neginfv);
        let mut ipv = _mm_or_si128(_mm_slli_si128::<2>(imx[q - 1]), neginfv);

        for qi in 0..q {
            let vb = qi * 7; // interleaved transition-vector base for this q
            // M(i,q): max(B->M, M->M, I->M, D->M) + emission. (vitfilter.c:145-150)
            let mut sv = _mm_adds_epi16(xbv, ld(twp, vb)); // BM
            sv = _mm_max_epi16(sv, _mm_adds_epi16(mpv, ld(twp, vb + 1))); // MM
            sv = _mm_max_epi16(sv, _mm_adds_epi16(ipv, ld(twp, vb + 2))); // IM
            sv = _mm_max_epi16(sv, _mm_adds_epi16(dpv, ld(twp, vb + 3))); // DM
            sv = _mm_adds_epi16(sv, ld(rwp, rbase + qi)); // emission
            xev = _mm_max_epi16(xev, sv);

            // Reload prev-row {MDI}(i-1,q) before the delayed stores. (vitfilter.c:155-161)
            mpv = mmx[qi];
            dpv = dmx[qi];
            ipv = imx[qi];
            mmx[qi] = sv;
            dmx[qi] = dcv;

            // Partial next D(i,q+1): M->D only, delayed in dcv. (vitfilter.c:166-167)
            dcv = _mm_adds_epi16(sv, ld(twp, vb + 4)); // MD
            dmaxv = _mm_max_epi16(dcv, dmaxv);

            // I(i,q). (vitfilter.c:170-171)
            let sv_i = _mm_adds_epi16(mpv, ld(twp, vb + 5)); // MI
            imx[qi] = _mm_max_epi16(sv_i, _mm_adds_epi16(ipv, ld(twp, vb + 6))); // II
        }

        // Specials (vitfilter.c:175-181). Identical to the scalar oracle.
        let xe = hmax_epi16(xev);
        if xe >= 32767 {
            return None; // eslERANGE
        }
        xn = adds(xn, 0);
        xc = xc.max(adds(xe, f.xw_e_move));
        xj = xj.max(adds(xe, f.xw_e_loop));
        xb = adds(xj, xw_move).max(adds(xn, xw_move));

        // Lazy-F DD loop (vitfilter.c:197-231).
        let dmax = hmax_epi16(dmaxv);
        if (dmax as i32) + (f.ddbound_w as i32) > (xb as i32) {
            dcv = _mm_or_si128(_mm_slli_si128::<2>(dcv), neginfv);
            for qi in 0..q {
                let d = _mm_max_epi16(dcv, dmx[qi]);
                dmx[qi] = d;
                dcv = _mm_adds_epi16(d, ld(twp, dd_base + qi));
            }
            // Up to three more passes; stop when a full pass adds nothing.
            loop {
                dcv = _mm_or_si128(_mm_slli_si128::<2>(dcv), neginfv);
                let mut qi = 0;
                while qi < q {
                    if _mm_movemask_epi8(_mm_cmpgt_epi16(dcv, dmx[qi])) == 0 {
                        break;
                    }
                    let d = _mm_max_epi16(dcv, dmx[qi]);
                    dmx[qi] = d;
                    dcv = _mm_adds_epi16(d, ld(twp, dd_base + qi));
                    qi += 1;
                }
                if qi != q {
                    break;
                }
            }
        } else {
            // Not calculating DD: just store the last M->D vector. (vitfilter.c:229-230)
            dmx[0] = _mm_or_si128(_mm_slli_si128::<2>(dcv), neginfv);
        }
    }

    // C->T (vitfilter.c:239-247).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p7_hmm::P7Profile;

    /// Tiny deterministic LCG (Numerical Recipes) — no external RNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        /// Uniform f32 in [0.05, 1.05) — kept strictly positive so all log-scores
        /// are finite (build_vit_filter takes ln of every used mat/trans entry).
        fn unit(&mut self) -> f32 {
            0.05 + (self.next_u32() as f32 / u32::MAX as f32)
        }
    }

    /// Build a random-but-valid RNA p7 filter profile of length M.
    fn random_profile(rng: &mut Lcg, m: i32) -> P7Profile {
        let mut p = P7Profile::new(m);
        for k in 1..=m as usize {
            let mut s = 0.0f32;
            let mut e = [0.0f32; 4];
            for x in 0..4 {
                e[x] = rng.unit();
                s += e[x];
            }
            for x in 0..4 {
                e[x] /= s;
            }
            p.mat[k] = e;
        }
        // Transitions [MM,MI,MD,IM,II,DM,DD], normalized within each source state.
        for k in 0..=m as usize {
            let (mm, mi, md) = (rng.unit(), rng.unit(), rng.unit());
            let ms = mm + mi + md;
            let (im, ii) = (rng.unit(), rng.unit());
            let is = im + ii;
            let (dm, dd) = (rng.unit(), rng.unit());
            let ds = dm + dd;
            p.trans[k] = [mm / ms, mi / ms, md / ms, im / is, ii / is, dm / ds, dd / ds];
        }
        p
    }

    fn random_dsq(rng: &mut Lcg, l: usize) -> Vec<u8> {
        // Index 0 sentinel; 1..=l residues; last sentinel. Codes 0..=15 exercise
        // canonical + IUPAC-degenerate + gap emission rows of the striped table.
        let mut dsq = vec![0u8; l + 2];
        for i in 1..=l {
            dsq[i] = (rng.next_u32() % 16) as u8;
        }
        dsq
    }

    /// The striped-SSE kernel must be byte-identical to the scalar oracle across
    /// every Q=ceil(M/8) boundary, a range of sequence lengths, and many random
    /// profiles/sequences. Compares raw f32 bits (so any xC divergence is caught).
    #[test]
    fn sse_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("sse2") {
                return; // no SSE2 -> dispatcher uses scalar; nothing to diff.
            }
            let mut rng = Lcg(0x1234_5678_9abc_def0);
            // M values straddling the 8-word vector boundary (and the Q>=2 floor).
            let ms = [1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 40, 63, 64, 65, 100, 128, 200];
            let ls = [1usize, 2, 5, 17, 50, 200, 500];
            let mut n_checked = 0u64;
            for &m in &ms {
                for _rep in 0..8 {
                    let p = random_profile(&mut rng, m);
                    let vf = build_vit_filter(&p);
                    for &l in &ls {
                        let dsq = random_dsq(&mut rng, l);
                        let sc_scalar = vit_score_scalar(&vf, &dsq, l);
                        let sc_sse = unsafe { vit_score_sse(&vf, &dsq, l) };
                        let bits =
                            |o: Option<f32>| o.map(|v| v.to_bits());
                        assert_eq!(
                            bits(sc_scalar),
                            bits(sc_sse),
                            "VF mismatch M={m} L={l}: scalar={sc_scalar:?} sse={sc_sse:?}"
                        );
                        n_checked += 1;
                    }
                }
            }
            assert!(n_checked > 1000, "expected many checks, got {n_checked}");
        }
    }
}
