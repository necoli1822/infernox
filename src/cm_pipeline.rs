//! Faithful 1:1 port of C Infernal's search pipeline (`original/src/cm_pipeline.c`)
//! and the HMMER MSV/SSV filter machinery it calls.
//!
//! METHODOLOGY (project memory `methodology-c-parity-before-speed`): transcribe C's
//! exact constants, data structures, and arithmetic — do NOT approximate. Verify
//! byte-for-byte against C (golden) before optimizing. Stage map:
//! `tests/parity/C_PIPELINE_MAP.md`.
//!
//! Stage cascade (standard pass): F1 MSV → F3 local Forward → F4/F5 glocal envelope
//! definition → F6 HMM-banded CYK filter → F7 Inside final scoring.

use crate::p7_hmm::P7Profile;
use crate::easel::exponential::esl_exp_surv;
use crate::easel::gumbel::esl_gumbel_invsurv;

/// Per-search pipeline configuration — mirrors C's `CM_PIPELINE` threshold/flag
/// fields (`cm_pipeline.c` struct + `cm_pipeline_Create`). Thresholds are P-value
/// cutoffs applied as `if (P > threshold) { reject }`.
#[derive(Debug, Clone)]
pub struct CmPipeline {
    pub f1: f64,
    pub f1b: f64,
    pub f2: f64,
    pub f2b: f64,
    pub f3: f64,
    pub f3b: f64,
    pub f4: f64,
    pub f4b: f64,
    pub f5: f64,
    pub f5b: f64,
    pub f6: f64,

    pub do_msv: bool,
    pub do_msvbias: bool,
    pub do_vit: bool,
    pub do_vitbias: bool,
    pub do_fwd: bool,
    pub do_fwdbias: bool,
    pub do_gfwd: bool,
    pub do_gfwdbias: bool,
    pub do_edef: bool,
    pub do_edefbias: bool,
    pub do_fcyk: bool,

    pub z: f64,
}

impl CmPipeline {
    /// Mirror of `cm_pipeline_Create` default-threshold selection by database size.
    /// `z_residues` is total searched residues (both strands). C src/cm_pipeline.c:510-560
    /// (`Z_Mb = Z / 1000000.`, `eslSMALLX1` is a tiny epsilon):
    /// ```c
    /// if(Z_Mb >= (20000. - eslSMALLX1)) { /* Z >= 20 Gb */
    ///   pli->F1=0.06; pli->F2=pli->F2b=0.02; pli->F3=pli->F3b=0.0002; pli->F4=pli->F4b=0.0002; pli->F5=pli->F5b=0.0002; pli->F6=0.0001;
    /// } else if(Z_Mb >= (2000. - eslSMALLX1)) { /* 20 Gb > Z >= 2 Gb */
    ///   pli->F1=0.15; pli->F2=pli->F2b=0.15; pli->F3=pli->F3b=0.0002; pli->F4=pli->F4b=0.0002; pli->F5=pli->F5b=0.0002; pli->F6=0.0001;
    /// } else if(Z_Mb >= (200. - eslSMALLX1)) { /* 2 Gb > Z >= 200 Mb */
    ///   pli->F1=0.15; pli->F2=pli->F2b=0.15; pli->F3=pli->F3b=0.0008; pli->F4=pli->F4b=0.0008; pli->F5=pli->F5b=0.0008; pli->F6=0.0001;
    /// } else if(Z_Mb >= (20. - eslSMALLX1)) { /* 200 Mb > Z >= 20 Mb */
    ///   pli->F1=pli->F1b=0.35; pli->F2=pli->F2b=0.15; pli->F3=pli->F3b=0.003; pli->F4=pli->F4b=0.003; pli->F5=pli->F5b=0.003; pli->F6=0.0001;
    /// } else if(Z_Mb >= (2. - eslSMALLX1)) { /* 20 Mb > Z >= 2 Mb */
    ///   pli->F1=0.35; pli->do_vit=pli->do_vitbias=FALSE; pli->F2=pli->F2b=1.00; pli->F3=pli->F3b=0.005; pli->F4=pli->F4b=0.005; pli->F5=pli->F5b=0.005; pli->F6=0.0001;
    /// } else { /* 2 Mb > Z */
    ///   pli->F1=0.35; pli->do_vit=pli->do_vitbias=FALSE; pli->F2=pli->F2b=1.00; pli->F3=pli->F3b=0.02; pli->F4=pli->F4b=0.02; pli->F5=pli->F5b=0.02; pli->F6=0.0001;
    /// }
    /// ```
    pub fn new_default(z_residues: f64) -> Self {
        let z_mb = z_residues / 1.0e6;
        let mut p = CmPipeline {
            f1: 0.35, f1b: 1.0,
            f2: 1.0, f2b: 1.0,
            f3: 0.02, f3b: 0.02,
            f4: 0.02, f4b: 0.02,
            f5: 0.02, f5b: 1.0,
            f6: 0.0001,
            do_msv: true, do_msvbias: false,
            do_vit: true, do_vitbias: true,
            do_fwd: true, do_fwdbias: true,
            do_gfwd: true, do_gfwdbias: true,
            do_edef: true, do_edefbias: false,
            do_fcyk: true,
            z: z_residues,
        };
        if z_mb >= 20_000.0 {
            p.f1 = 0.06; p.f2 = 0.02; p.f2b = 0.02;
            p.f3 = 0.0002; p.f3b = 0.0002;
            p.f4 = 0.0002; p.f4b = 0.0002; p.f5 = 0.0002;
        } else if z_mb >= 2_000.0 {
            p.f1 = 0.15; p.f2 = 0.15; p.f2b = 0.15;
            p.f3 = 0.0002; p.f3b = 0.0002;
            p.f4 = 0.0002; p.f4b = 0.0002; p.f5 = 0.0002;
        } else if z_mb >= 200.0 {
            p.f1 = 0.15; p.f2 = 0.15; p.f2b = 0.15;
            p.f3 = 0.0008; p.f3b = 0.0008;
            p.f4 = 0.0008; p.f4b = 0.0008; p.f5 = 0.0008;
        } else if z_mb >= 20.0 {
            p.f1 = 0.35; p.f2 = 0.15; p.f2b = 0.15;
            p.f3 = 0.003; p.f3b = 0.003;
            p.f4 = 0.003; p.f4b = 0.003; p.f5 = 0.003;
        } else if z_mb >= 2.0 {
            p.f1 = 0.35;
            p.do_vit = false; p.do_vitbias = false; p.f2 = 1.0; p.f2b = 1.0;
            p.f3 = 0.005; p.f3b = 0.005;
            p.f4 = 0.005; p.f4b = 0.005; p.f5 = 0.005;
        } else {
            p.f1 = 0.35;
            p.do_vit = false; p.do_vitbias = false; p.f2 = 1.0; p.f2b = 1.0;
            p.f3 = 0.02; p.f3b = 0.02;
            p.f4 = 0.02; p.f4b = 0.02; p.f5 = 0.02;
        }
        p
    }
}

// ============================================================================
// F1: MSV/SSV filter — faithful transcription of HMMER's quantized MSV machinery.
// Sources: impl_sse/p7_oprofile.c (mf_conversion, byteify), impl_sse/msvfilter.c
// (p7_SSVFilter_longtarget), p7_scoredata.c (ssv_scores, prefix/suffix_lengths),
// p7_pipeline.c (p7_pli_ExtendAndMergeWindows), modelconfig.c (match scores).
// ============================================================================

const KP: usize = 18; // RNA extended alphabet size (A,C,G,U,gap,degens,*,~)
const K_CANON: usize = 4;
const WINDOW_BETA: f64 = 1e-7; // p7_DEFAULT_WINDOW_BETA

/// RNA IUPAC degeneracy: the canonical residues (0=A,1=C,2=G,3=U) that each code
/// 0..Kp-1 expands to. Empty = gap(4)/nonresidue(16)/missing(17) → score -inf.
/// Matches `easel/src/alphabet.rs` rna() inmap.
fn degen_set(code: usize) -> &'static [usize] {
    match code {
        0 => &[0], 1 => &[1], 2 => &[2], 3 => &[3],
        5 => &[0, 2],       // R = A|G
        6 => &[1, 3],       // Y = C|U
        7 => &[0, 1],       // M = A|C
        8 => &[2, 3],       // K = G|U
        9 => &[1, 2],       // S = C|G
        10 => &[0, 3],      // W = A|U
        11 => &[0, 1, 3],   // H = A|C|U
        12 => &[1, 2, 3],   // B = C|G|U
        13 => &[0, 1, 2],   // V = A|C|G
        14 => &[0, 2, 3],   // D = A|G|U
        15 => &[0, 1, 2, 3],// N = any
        _ => &[],
    }
}

/// C `unbiased_byteify` (p7_oprofile.c:527-536), for transition/length costs:
/// ```c
/// static uint8_t unbiased_byteify(P7_OPROFILE *om, float sc) {
///   uint8_t b;
///   sc  = -1.0f * roundf(om->scale_b * sc);          /* ugh. sc is now an integer cost represented in a float... */
///   b   = (sc > 255.) ? 255 : (uint8_t) sc;          /* and now we cast, saturate, and bias it to an unsigned char cost... */
///   return b;
/// }
/// ```
#[inline]
fn unbiased_byteify(scale_b: f32, sc: f32) -> u8 {
    let v = -(scale_b * sc).round();
    if v >= 255.0 { 255 } else if v <= 0.0 { 0 } else { v as u8 }
}

/// C `biased_byteify` (p7_oprofile.c:504-514), for match emission costs:
/// ```c
/// static uint8_t biased_byteify(P7_OPROFILE *om, float sc) {
///   uint8_t b;
///   sc  = -1.0f * roundf(om->scale_b * sc);                     /* ugh. sc is now an integer cost represented in a float... */
///   b   = (sc > 255 - om->bias_b) ? 255 : (uint8_t) sc + om->bias_b;  /* and now we cast and saturate it to an unsigned char cost... */
///   return b;
/// }
/// ```
#[inline]
fn biased_byteify(scale_b: f32, bias_b: u8, sc: f32) -> u8 {
    let v = -(scale_b * sc).round();
    if v > (255 - bias_b as i32) as f32 {
        255
    } else {
        (v as i32 + bias_b as i32).clamp(0, 255) as u8
    }
}

/// Quantized MSV filter profile + the P7_SCOREDATA fields the SSV filter needs.
pub struct MsvFilter {
    pub m: usize,
    scale_b: f32,
    base_b: u8,
    bias_b: u8,
    tbm_b: u8,
    tec_b: u8,
    tjb_b: u8,
    /// Byteified emission costs `rbv[k][x]` (== unstriped `om->rbv` == `ssv_scores`).
    rbv: Vec<[u8; KP]>,
    /// Transposed emission costs for SIMD: rbv_t[x*(m+1)+k] = rbv[k][x].
    rbv_t: Vec<u8>,
    mmu: f64,
    mlambda: f64,
    /// `data->prefix_lengths` (cumulative) and `data->suffix_lengths` from p7_scoredata.c.
    prefix: Vec<f32>,
    suffix: Vec<f32>,
    pub max_length: usize,
}

/// Build the quantized MSV filter from the P7 filter HMM (`mf_conversion` +
/// `p7_hmm_ScoreDataComputeRest`). `max_length` is `om->max_length` = `pli->maxW`.
pub fn build_msv_filter(p7: &P7Profile, max_length: usize) -> MsvFilter {
    let m = p7.m as usize;
    // C mf_conversion (p7_oprofile.c:551-553):
    //   om->scale_b = 3.0 / eslCONST_LOG2;   /* scores in units of third-bits */
    //   om->base_b  = 190;
    let scale_b = (3.0_f64 / std::f64::consts::LN_2) as f32;
    let base_b: u8 = 190;

    // Per-position match scores sc[k][x] = log-odds vs bg (modelconfig.c:142-151;
    // bg->f[x]=1/K=0.25; ISC hardwired to 0 which participates in maxsc below):
    //   for (x = 0; x < hmm->abc->K; x++)  sc[x] = log((double)hmm->mat[k][x] / bg->f[x]);
    //   esl_abc_FExpectScVec(hmm->abc, sc, bg->f);
    let mut msc = vec![[f32::NEG_INFINITY; KP]; m + 1];
    let mut maxsc = 0.0f32;
    for k in 1..=m {
        let mut sc = [f32::NEG_INFINITY; KP];
        for x in 0..K_CANON {
            sc[x] = ((p7.mat[k][x] as f64) / 0.25).ln() as f32;
            if sc[x] > maxsc {
                maxsc = sc[x];
            }
        }
        // Degenerate residues via esl_abc_FExpectScVec: expected score over the
        // canonical residues in the code, weighted by bg (uniform → simple mean).
        for code in K_CANON..KP {
            let set = degen_set(code);
            if set.is_empty() {
                continue; // gap/*/~ stay -inf
            }
            let mut s = 0.0f32;
            for &x in set {
                s += sc[x];
            }
            sc[code] = s / set.len() as f32;
        }
        msc[k] = sc;
    }

    // C mf_conversion (p7_oprofile.c:556-565): bias from the max match score, then
    // byteify the emission cost vector rbv:
    //   om->bias_b = unbiased_byteify(om, -1.0 * maxval);
    //   for (x = 0; x < gm->abc->Kp; x++)
    //     for (k = 1, q = 0; q < nq; q++, k++)  { ... rbv ... biased_byteify(om, p7P_MSC(gm,k+..,x)) }
    let bias_b = unbiased_byteify(scale_b, -maxsc);
    let mut rbv = vec![[255u8; KP]; m + 1];
    for k in 1..=m {
        for x in 0..KP {
            rbv[k][x] = biased_byteify(scale_b, bias_b, msc[k][x]);
        }
    }
    // Transposed copy for the SIMD SSV row: rbv_t[x*(m+1)+k] = rbv[k][x].
    let mut rbv_t = vec![0u8; KP * (m + 1)];
    for k in 0..=m {
        for x in 0..KP {
            rbv_t[x * (m + 1) + k] = rbv[k][x];
        }
    }

    // C mf_conversion (p7_oprofile.c:568-570), special-state transition costs:
    //   om->tbm_b = unbiased_byteify(om, logf(2.0f / ((float) gm->M * (float) (gm->M+1))));  /* B->Mk */
    //   om->tec_b = unbiased_byteify(om, logf(0.5f));                                        /* E->C  */
    //   om->tjb_b = unbiased_byteify(om, logf(3.0f / (float) (om->max_length+3)));           /* J->B  */
    let tbm_b = unbiased_byteify(scale_b, (2.0_f64 / (m as f64 * (m as f64 + 1.0))).ln() as f32);
    let tec_b = unbiased_byteify(scale_b, 0.5_f32.ln());
    let tjb_b = unbiased_byteify(scale_b, (3.0_f64 / (max_length as f64 + 3.0)).ln() as f32);

    if std::env::var("DUMP_MSV").is_ok() {
        eprintln!(
            "[R MSV] M={} max_length={} scale_b={:.6} base_b={} bias_b={} tbm_b={} tec_b={} tjb_b={}",
            m, max_length, scale_b, base_b, bias_b, tbm_b, tec_b, tjb_b
        );
        for kk in 1..=3.min(m) {
            for xx in 0..4 {
                eprintln!("[R MSC] k={} x={} MSC={:.6}", kk, xx, msc[kk][xx]);
            }
        }
    }

    // prefix/suffix window-extension lengths. C p7_hmm_ScoreDataComputeRest
    // (p7_scoredata.c:361-379); t_mis=fwd MI trans = trans[k][1], t_iis=II=trans[k][4]:
    //   sum = 0;
    //   for (k=1; k < om->M; k++) {
    //     if (t_mis[k] == 0)  data->prefix_lengths[k] = 1;
    //     else                data->prefix_lengths[k] = 1 + (int)(log(p7_DEFAULT_WINDOW_BETA / t_mis[k]) / log(t_iis[k]));
    //     sum += data->prefix_lengths[k];
    //   }
    //   data->prefix_lengths[0] = data->prefix_lengths[om->M] = 0;
    //   for (k=1; k < om->M; k++)  data->prefix_lengths[k] /= sum;
    //   data->suffix_lengths[om->M] = data->prefix_lengths[om->M-1];
    //   for (k=om->M-1; k >= 1; k--)  data->suffix_lengths[k] = data->suffix_lengths[k+1] + data->prefix_lengths[k-1];
    //   for (k=2; k < om->M; k++)     data->prefix_lengths[k] += data->prefix_lengths[k-1];
    let mut prefix = vec![0.0f32; m + 1];
    let mut suffix = vec![0.0f32; m + 1];
    let mut sum = 0.0f32;
    for k in 1..m {
        let t_mi = p7.trans[k][1] as f64;
        let t_ii = p7.trans[k][4] as f64;
        prefix[k] = if t_mi == 0.0 {
            1.0
        } else {
            let lii = t_ii.ln();
            if lii == 0.0 || !lii.is_finite() {
                1.0
            } else {
                (1.0 + ((WINDOW_BETA / t_mi).ln() / lii).trunc()) as f32
            }
        };
        sum += prefix[k];
    }
    prefix[0] = 0.0;
    prefix[m] = 0.0;
    if sum > 0.0 {
        for k in 1..m {
            prefix[k] /= sum;
        }
    }
    if m >= 1 {
        suffix[m] = if m >= 1 { prefix[m - 1] } else { 0.0 };
        for k in (1..m).rev() {
            suffix[k] = suffix[k + 1] + prefix[k - 1];
        }
        for k in 2..m {
            prefix[k] += prefix[k - 1];
        }
    }

    if std::env::var("DUMP_PSL").is_ok() {
        for k in 0..=m {
            eprintln!(
                "[R PSL] k={} prefix={:.6} suffix={:.6} t_mi={:.6} t_ii={:.6}",
                k, prefix[k], suffix[k], p7.trans[k][1], p7.trans[k][4]
            );
        }
    }

    MsvFilter {
        m,
        scale_b,
        base_b,
        bias_b,
        tbm_b,
        tec_b,
        tjb_b,
        rbv,
        rbv_t,
        mmu: p7.evparam.lmmu,
        mlambda: p7.evparam.lmlambda,
        prefix,
        suffix,
        max_length,
    }
}

impl MsvFilter {
    /// `(255 - om->base_b) / om->scale_b` — the MSV score assigned on eslERANGE
    /// overflow (evalues.c::p7_MSVMu:239).
    pub fn cal_maxsc(&self) -> f32 {
        (255.0 - self.base_b as f32) / self.scale_b
    }
    pub fn cal_scale_b(&self) -> f32 {
        self.scale_b
    }
    pub fn cal_base_b(&self) -> u8 {
        self.base_b
    }
}

/// Whole-sequence MSV score, a faithful scalar transcription of
/// `p7_MSVFilter` (impl_sse/msvfilter.c:83). Returns `None` on the eslERANGE
/// overflow condition (the SIMD ceiling test), matching the C caller which then
/// substitutes `maxsc = (255-base_b)/scale_b`.
///
/// The striped SSE recurrence and this natural-order scalar loop compute the
/// identical final `xJ` byte: MSV has no gaps between M cells in a row and only
/// max/saturating-add ops, so the byte result is layout-independent. C also tries
/// `p7_SSVFilter` first, but that returns the *identical* score whenever it
/// succeeds (single-hit == multihit when J is not beneficial), so this covers
/// both paths.
///
/// `dsq` is 1-indexed with sentinels at `[0]` and `[L+1]`. The filter's `tjb_b`
/// must already be configured for length `L` (build via `build_msv_filter(p7, L)`).
///
/// ```c
/// xJv = subs(biasv,biasv)=0;  xBv = subs(basev, tjbmv);   // tjbm = tjb_b + tbm_b
/// for (i=1..L) {
///   xEv=0; mpv = slli(dp[Q-1]);   // -inf in shifted lane
///   for (q) { sv = max(mpv,xBv); sv = adds(sv,biasv); sv = subs(sv,rsc); xEv=max(xEv,sv); mpv=dp[q]; dp[q]=sv; }
///   if (adds(xEv,biasv) == 0xFF) return eslERANGE;
///   xEv = subs(xEv,tecv); xJv = max(xJv,xEv); xBv = subs(max(basev,xJv), tjbmv);
/// }
/// ret = ((float)(xJ - tjb_b) - (float)base_b) / scale_b - 3.0;
/// ```
pub fn msv_score(f: &MsvFilter, dsq: &[u8], l: usize) -> Option<f32> {
    msv_score_tjb(f, dsq, l, f.tjb_b)
}

/// Full MSV score for a window whose MSV length model has been reconfigured to
/// `wlen` residues (C `p7_oprofile_ReconfigMSVLength(om, wlen)`,
/// p7_oprofile.c: `om->tjb_b = unbiased_byteify(om, logf(3.0/(L+3)))`). Used by the
/// F1b MSV composition-bias sub-filter, which reruns the per-sequence MSV filter on
/// each surviving window with that window's own length (cm_pipeline.c:2669-2670).
pub fn msv_score_reconfig(f: &MsvFilter, dsq: &[u8], l: usize, wlen: usize) -> Option<f32> {
    let tjb_b = unbiased_byteify(f.scale_b, (3.0_f64 / (wlen as f64 + 3.0)).ln() as f32);
    msv_score_tjb(f, dsq, l, tjb_b)
}

/// Core striped-MSV score with an explicit `tjb_b` (J->B move cost), so callers can
/// reconfigure the MSV length model per window without rebuilding the filter.
fn msv_score_tjb(f: &MsvFilter, dsq: &[u8], l: usize, tjb_b: u8) -> Option<f32> {
    let m = f.m;
    let bias = f.bias_b;
    let tec = f.tec_b;
    let base = f.base_b;
    // tjbmv = set1_epi8((int8_t)tjb_b + (int8_t)tbm_b): low byte of the sum.
    let tjbm = tjb_b.wrapping_add(f.tbm_b);

    // dp[k] holds M(i-1,k); dp[0] unused (the striped right-shift injects -inf=0
    // as M(i-1,0)). Initialized to 0 = -infinity in offset arithmetic.
    let mut dp = vec![0u8; m + 1];
    let mut xj: u8 = 0;
    let mut xb: u8 = base.saturating_sub(tjbm);

    for i in 1..=l {
        let x = dsq[i] as usize;
        let mut xe: u8 = 0;
        let mut mpv: u8 = 0; // M(i-1, 0) = -inf
        let rbv_row = &f.rbv[0..m + 1];
        for k in 1..=m {
            let mut sv = mpv.max(xb);
            sv = sv.saturating_add(bias);
            sv = sv.saturating_sub(rbv_row[k][x]);
            if sv > xe {
                xe = sv;
            }
            mpv = dp[k];
            dp[k] = sv;
        }
        // Overflow test: any lane hitting the 0xFF ceiling after adding bias.
        if xe.saturating_add(bias) == 255 {
            return None; // eslERANGE
        }
        xe = xe.saturating_sub(tec);
        xj = xj.max(xe);
        xb = base.max(xj).saturating_sub(tjbm);
    }

    let ret = ((xj as i32 - tjb_b as i32) as f32 - base as f32) / f.scale_b - 3.0;
    Some(ret)
}

/// A raw SSV diagonal (before window extension): `n`=target start, `length`=model
/// diagonal length, `k`=model end position, `score`=diagonal bit score.
#[derive(Debug, Clone, Copy)]
struct RawWin {
    n: i64,
    length: i64,
    k: i64,
    score: f32,
}

/// A candidate window passed to later pipeline stages. 1-based inclusive sequence
/// coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Window {
    pub start: i64,
    pub end: i64,
    pub score: f32,
}

/// F1: faithful scalar transcription of `p7_SSVFilter_longtarget`
/// (impl_sse/msvfilter.c:256). Same uint8 saturating recurrence as the SIMD code
/// (striping is only a memory layout), so this produces byte-identical diagonals.
/// AVX2 32-wide SSV row: `new[k] = subs(adds(max(dp[k-1], xb), bias), costx[k])` for
/// the leading 32-aligned span, updating `*k` to the first unprocessed index and
/// returning the partial byte max. Element-wise saturating u8 (no cross-lane carry)
/// so this is BIT-IDENTICAL to the 16-wide SSE / scalar versions — the wider vector
/// only changes how many lanes are done per step, not any lane's value. The SSE
/// 16-wide loop + scalar tail in the caller finish the [ *k .. m ] remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ssv_row_avx2(
    dp: &[u8],
    new: &mut [u8],
    costx: &[u8],
    xb: u8,
    bias: u8,
    m: usize,
    k: &mut usize,
) -> u8 {
    use std::arch::x86_64::*;
    let xbv = _mm256_set1_epi8(xb as i8);
    let biasv = _mm256_set1_epi8(bias as i8);
    let mut maxv = _mm256_setzero_si256();
    let mut kk = *k;
    while kk + 32 <= m + 1 {
        let mpv = _mm256_loadu_si256(dp.as_ptr().add(kk - 1) as *const __m256i);
        let mut sv = _mm256_max_epu8(mpv, xbv);
        sv = _mm256_adds_epu8(sv, biasv);
        let cv = _mm256_loadu_si256(costx.as_ptr().add(kk) as *const __m256i);
        sv = _mm256_subs_epu8(sv, cv);
        _mm256_storeu_si256(new.as_mut_ptr().add(kk) as *mut __m256i, sv);
        maxv = _mm256_max_epu8(maxv, sv);
        kk += 32;
    }
    *k = kk;
    let mut tmp = [0u8; 32];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, maxv);
    let mut xm = 0u8;
    for &v in tmp.iter() {
        if v > xm {
            xm = v;
        }
    }
    xm
}

/// `dsq` is 1-indexed with sentinels; residues at `1..=l`.
fn ssv_diagonals(f: &MsvFilter, dsq: &[u8], l: usize, f1: f64) -> Vec<RawWin> {
    let ln2 = std::f64::consts::LN_2;
    let p1 = f.max_length as f64 / (f.max_length as f64 + 1.0);
    let nullsc = f.max_length as f64 * p1.ln() + (1.0 - p1).ln();
    let invp = esl_gumbel_invsurv(f1, f.mmu, f.mlambda);
    // sc_thresh. C msvfilter.c:324:
    //   sc_thresh = (int) ceil( ( ( nullsc + (invP * eslCONST_LOG2) + 3.0 ) * om->scale_b )
    //                           + om->base_b + om->tec_b + om->tjb_b );
    let sc_thresh = ((nullsc + invp * ln2 + 3.0) * f.scale_b as f64
        + f.base_b as f64
        + f.tec_b as f64
        + f.tjb_b as f64)
        .ceil() as i32;

    // xB (B state). C msvfilter.c:337-340:
    //   basev = _mm_set1_epi8((int8_t) om->base_b);
    //   tjbmv = _mm_set1_epi8((int8_t) om->tjb_b + (int8_t) om->tbm_b);
    //   xBv   = _mm_subs_epu8(basev, tjbmv);   // saturating
    let xb = f.base_b.saturating_sub(f.tjb_b.saturating_add(f.tbm_b));
    let m = f.m;
    // dp[k] = M(i-1,k); 0 == -inf in offset arithmetic. Reused across rows.
    let mut dp = vec![0u8; m + 1];
    let mut new = vec![0u8; m + 1];
    let mut wins = Vec::new();
    // Detect AVX2 once (not per row); the 32-wide SSV span is bit-identical.
    #[cfg(target_arch = "x86_64")]
    let ssv_use_avx2 = is_x86_feature_detected!("avx2");

    let cost = |k: usize, code: usize| -> u8 {
        if code < KP {
            f.rbv[k][code]
        } else {
            255
        }
    };

    let mut i = 1usize;
    while i <= l {
        let x = dsq[i] as usize;
        // SSV row recurrence. C msvfilter.c:352-361 (striped SIMD). Each new[k]
        // depends ONLY on the previous row (dp[k-1]) + scalar xB, so 16-wide SSE
        // uint8 (max/adds/subs) is bit-identical per byte to the scalar version.
        // rbv_t gives the per-k emission cost of residue x contiguously.
        //   sv = subs_epu8(adds_epu8(max_epu8(dp[k-1], xB), bias), rbv[k][x])
        let costx = &f.rbv_t[x * (m + 1)..];
        let xe: u8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let mut xmax = 0u8;
            let mut k = 1usize;
            // AVX2 32-wide leading span (bit-identical to SSE/scalar; see ssv_row_avx2).
            if ssv_use_avx2 {
                let m2 = ssv_row_avx2(&dp, &mut new, costx, xb, f.bias_b, m, &mut k);
                if m2 > xmax {
                    xmax = m2;
                }
            }
            // SSE 16-wide remainder.
            let xbv = _mm_set1_epi8(xb as i8);
            let biasv = _mm_set1_epi8(f.bias_b as i8);
            let mut maxv = _mm_setzero_si128();
            while k + 16 <= m + 1 {
                let mpv = _mm_loadu_si128(dp.as_ptr().add(k - 1) as *const __m128i);
                let mut sv = _mm_max_epu8(mpv, xbv);
                sv = _mm_adds_epu8(sv, biasv);
                let cv = _mm_loadu_si128(costx.as_ptr().add(k) as *const __m128i);
                sv = _mm_subs_epu8(sv, cv);
                _mm_storeu_si128(new.as_mut_ptr().add(k) as *mut __m128i, sv);
                maxv = _mm_max_epu8(maxv, sv);
                k += 16;
            }
            // horizontal max of maxv, folded into the AVX2 partial max
            let mut tmp = [0u8; 16];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, maxv);
            for &v in tmp.iter() { if v > xmax { xmax = v; } }
            // scalar tail
            while k <= m {
                let sv = dp[k - 1].max(xb).saturating_add(f.bias_b).saturating_sub(costx[k]);
                new[k] = sv;
                if sv > xmax { xmax = sv; }
                k += 1;
            }
            xe = xmax;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut xmax = 0u8;
            for k in 1..=m {
                let sv = dp[k - 1].max(xb).saturating_add(f.bias_b).saturating_sub(costx[k]);
                new[k] = sv;
                if sv > xmax { xmax = sv; }
            }
            xe = xmax;
        }
        std::mem::swap(&mut dp, &mut new);

        // end_k (argmax model position) only needed when the row triggers; recover
        // it with a scalar rescan of dp (== new), matching the strict-`>` leftmost
        // argmax the original scalar loop produced (msvfilter.c:373-381).
        let mut end_k = 0usize;
        if (xe as i32) >= sc_thresh {
            let mut xe2 = 0u8;
            for k in 1..=m {
                if dp[k] > xe2 {
                    xe2 = dp[k];
                    end_k = k;
                }
            }
        }

        if (xe as i32) >= sc_thresh && end_k > 0 {
            if std::env::var("DUMP_TRIG").is_ok() {
                eprintln!("[R trig] i={} end={} rem_sc={}", i, end_k, xe);
            }
            // Recover the diagonal backward. C msvfilter.c:385-395:
            //   start = end; target_end = target_start = i; sc = rem_sc;
            //   while (rem_sc > om->base_b - om->tjb_b - om->tbm_b) {
            //     rem_sc -= om->bias_b - ssvdata->ssv_scores[start*Kp + dsq[target_start]];
            //     --start; --target_start; }
            //   start++; target_start++;
            let baseline = f.base_b as i32 - f.tjb_b as i32 - f.tbm_b as i32;
            let mut start = end_k;
            let mut tstart = i;
            let mut rem = xe as i32;
            while rem > baseline && start >= 1 && tstart >= 1 {
                rem -= f.bias_b as i32 - cost(start, dsq[tstart] as usize) as i32;
                start -= 1;
                tstart -= 1;
            }
            start += 1;
            tstart += 1;

            // Extend the diagonal forward. C msvfilter.c:399-422:
            //   k = end+1; n = target_end+1; max_end = target_end; max_sc = sc; pos_since_max = 0;
            //   while (k<om->M && n<=L) {
            //     sc += om->bias_b - ssvdata->ssv_scores[k*Kp + dsq[n]];
            //     if (sc >= max_sc) { max_sc = sc; max_end = n; pos_since_max = 0; }
            //     else { pos_since_max++; if (pos_since_max == 5) break; }
            //     k++; n++; }
            //   end += (max_end - target_end); target_end = max_end;
            let mut sc2 = xe as i32;
            let mut max_sc = sc2;
            let mut max_end = i;
            let mut since = 0;
            let mut k = end_k + 1;
            let mut n = i + 1;
            while k < m && n <= l {
                sc2 += f.bias_b as i32 - cost(k, dsq[n] as usize) as i32;
                if sc2 >= max_sc {
                    max_sc = sc2;
                    max_end = n;
                    since = 0;
                } else {
                    since += 1;
                    if since == 5 {
                        break;
                    }
                }
                k += 1;
                n += 1;
            }

            let end_model = end_k + (max_end - i);
            // Diagonal bit score. C msvfilter.c:424-426:
            //   ret_sc = ((float) (max_sc - om->tjb_b) - (float) om->base_b);
            //   ret_sc /= om->scale_b;
            //   ret_sc -= 3.0;   // ~ L log(L/(L+3)), for NN,CC,JJ
            let ret_sc = ((max_sc - f.tjb_b as i32) as f32 - f.base_b as f32) / f.scale_b - 3.0;
            if std::env::var("DUMP_SSV").is_ok() {
                eprintln!(
                    "[R win] target_start={} target_end={} model_end={} diaglen={} max_sc={} ret_sc={:.4}",
                    tstart, max_end, end_model, end_model - start + 1, max_sc, ret_sc
                );
            }
            wins.push(RawWin {
                n: tstart as i64,
                length: (end_model - start + 1) as i64,
                k: end_model as i64,
                score: ret_sc,
            });

            // Reset dp to 0 (msvfilter.c:382), then skip forward: C sets
            // `i = target_end` (msvfilter.c:447) and the for-loop's `i++` resumes
            // at target_end+1 = max_end+1. So the just-found diagonal's residues
            // are NOT re-scanned. (A strong region can still emit several
            // diagonals, but only past the previous diagonal's end.)
            for v in dp.iter_mut() {
                *v = 0;
            }
            i = max_end + 1;
            continue;
        }
        i += 1;
    }
    wins
}

/// `p7_pli_ExtendAndMergeWindows` (hmmer/src/p7_pipeline.c:323). Extends each
/// diagonal into a window using prefix/suffix length estimates, then merges
/// overlapping windows. Forward strand only here (no FM complement handling).
fn extend_and_merge(f: &MsvFilter, mut raws: Vec<RawWin>, target_len: i64, pct_overlap: f32) -> Vec<Window> {
    if raws.is_empty() {
        return Vec::new();
    }
    // Extend windows. C p7_pipeline.c:356-364 (non-complement branch). NOTE: the
    // whole float expression is evaluated in double, THEN the int64_t assignment
    // truncates — so we must NOT pre-truncate the extension width.
    //   window_start = ESL_MAX( 1,                    curr_window->n -
    //                    (om->max_length * (0.1 + data->prefix_lengths[curr_window->k - curr_window->length + 1])) );
    //   window_end   = ESL_MIN( curr_window->target_len,  curr_window->n + curr_window->length +
    //                    (om->max_length * (0.1 + data->suffix_lengths[curr_window->k])) );
    //   curr_window->length = window_end - window_start + 1;
    //   curr_window->n = window_start;
    let maxl = f.max_length as f64;
    for w in raws.iter_mut() {
        let kstart = (w.k - w.length + 1).clamp(0, f.m as i64) as usize;
        let kend = (w.k).clamp(0, f.m as i64) as usize;
        let pref = f.prefix[kstart] as f64;
        let suff = f.suffix[kend] as f64;
        let window_start = (w.n as f64 - maxl * (0.1 + pref)).max(1.0) as i64;
        let window_end =
            (w.n as f64 + w.length as f64 + maxl * (0.1 + suff)).min(target_len as f64) as i64;
        w.n = window_start;
        w.length = window_end - window_start + 1;
    }

    // Merge overlapping windows, compressing in place. C p7_pipeline.c:368-394
    // (prev_window = windows+new_hit_cnt = last kept; same id/complementarity here):
    //   for (i=1; i<windowlist->count; i++) {
    //     prev_window = windowlist->windows+new_hit_cnt;
    //     curr_window = windowlist->windows+i;
    //     window_start = ESL_MAX(prev_window->n, curr_window->n);
    //     window_end   = ESL_MIN(prev_window->n+prev_window->length-1, curr_window->n+curr_window->length-1);
    //     window_len   = window_end - window_start + 1;
    //     if ( ... && (float)(window_len)/ESL_MIN(prev_window->length, curr_window->length) > pct_overlap ) {
    //       window_start = ESL_MIN(prev_window->n, curr_window->n);
    //       window_end   = ESL_MAX(prev_window->n+prev_window->length-1, curr_window->n+curr_window->length-1);
    //       prev_window->n = window_start;
    //       prev_window->length = window_end - window_start + 1;
    //     } else { new_hit_cnt++; windowlist->windows[new_hit_cnt] = windowlist->windows[i]; }
    //   }
    let mut kept: Vec<RawWin> = Vec::with_capacity(raws.len());
    kept.push(raws[0]);
    for i in 1..raws.len() {
        let curr = raws[i];
        let prev = kept.last_mut().unwrap();
        let window_start = prev.n.max(curr.n);
        let window_end = (prev.n + prev.length - 1).min(curr.n + curr.length - 1);
        let window_len = window_end - window_start + 1;
        let minlen = prev.length.min(curr.length);
        if (window_len as f32) / (minlen as f32) > pct_overlap {
            // merge: extend prev to span both
            let ms = prev.n.min(curr.n);
            let me = (prev.n + prev.length - 1).max(curr.n + curr.length - 1);
            prev.n = ms;
            prev.length = me - ms + 1;
        } else {
            kept.push(curr);
        }
    }

    kept.into_iter()
        .map(|w| Window {
            start: w.n,
            end: w.n + w.length - 1,
            score: w.score,
        })
        .collect()
}

/// Split windows longer than `2*cmw` into pieces of length `2*cmw` overlapping by
/// `cmw-1` residues. `cmw` is `pli->cmW` (the CM's W).
///
/// C src/cm_pipeline.c:2580-2612 (the `else` at 2608 = "do not split"):
/// ```c
/// /* split up windows > (2 * pli->cmW) into length 2W, with W-1
///  * overlapping residues. */
/// for (i = 0, i2 = 0; i < nwin; i++, i2++) {
///   wlen = we[i] - ws[i] + 1;
///   if(wlen > (2 * pli->cmW)) {
///     /* split this window */
///     new_ws[i2]   = ws[i];
///     new_we[i2]   = ESL_MIN((new_ws[i2] + (2 * pli->cmW) - 1), we[i]);
///     while(new_we[i2] < we[i]) {
///       i2++;
///       new_ws[i2]   = ESL_MIN(new_ws[i2-1] + pli->cmW, we[i]);
///       new_we[i2]   = ESL_MIN(new_we[i2-1] + pli->cmW, we[i]);
///     }
///   }
///   else { /* do not split this window */
///     new_ws[i2] = ws[i];
///     new_we[i2] = we[i];
///   }
/// }
/// ```
fn split_windows(wins: Vec<Window>, cmw: i64) -> Vec<Window> {
    let mut out: Vec<Window> = Vec::with_capacity(wins.len());
    for w in wins {
        let wlen = w.end - w.start + 1;
        if wlen > 2 * cmw {
            // new_ws[i2] = ws[i]; new_we[i2] = MIN(ws[i]+2*cmW-1, we[i]);
            let mut nws = w.start;
            let mut nwe = (w.start + 2 * cmw - 1).min(w.end);
            out.push(Window { start: nws, end: nwe, score: w.score });
            // while(new_we < we) { new_ws = MIN(prev_ws+cmW, we); new_we = MIN(prev_we+cmW, we); }
            while nwe < w.end {
                nws = (nws + cmw).min(w.end);
                nwe = (nwe + cmw).min(w.end);
                out.push(Window { start: nws, end: nwe, score: w.score });
            }
        } else {
            out.push(w);
        }
    }
    out
}

/// F1 stage on a single already-extracted window (`dsq` 1-indexed with sentinels,
/// window length `l`, local coordinates). `pct_overlap = 0.0` (cm_pipeline.c:2569).
/// `cmw` is `pli->cmW` (the CM's W); windows are split as C does post-merge.
pub fn f1_msv_windows(f: &MsvFilter, dsq: &[u8], l: usize, f1: f64, cmw: i64) -> Vec<Window> {
    let dbg = std::env::var("IX_DEBUG").is_ok();
    let t0 = std::time::Instant::now();
    let raws = ssv_diagonals(f, dsq, l, f1);
    if dbg { eprintln!("[MSV] ssv_diagonals -> {} raws  t={:.2}s (l={} cmw={})", raws.len(), t0.elapsed().as_secs_f64(), l, cmw); }
    let t1 = std::time::Instant::now();
    let merged = extend_and_merge(f, raws, l as i64, 0.0);
    if dbg { eprintln!("[MSV] extend_and_merge -> {} merged  t={:.2}s", merged.len(), t1.elapsed().as_secs_f64()); }
    let t2 = std::time::Instant::now();
    let sw = split_windows(merged, cmw);
    if dbg { eprintln!("[MSV] split_windows -> {} wins  t={:.2}s", sw.len(), t2.elapsed().as_secs_f64()); }
    sw
}

/// C's `CM_MAX_RESIDUE_COUNT` (src/infernal.h:116): residues of NEW sequence per
/// window read by `esl_sqio_ReadWindow`.
pub const CM_MAX_RESIDUE_COUNT: usize = 100_000;

/// Search-driver windowing. C serial_loop (cmsearch.c:810) reads the sequence with
///   `esl_sqio_ReadWindow(dbfp, pli->maxW, CM_MAX_RESIDUE_COUNT, dbsq)`
/// and runs the pipeline per window. The forward-strand coordinate math is
/// esl_sqio_ascii.c:1154-1160 (`C = pli->maxW`, `W = CM_MAX_RESIDUE_COUNT`, `sq->end`
/// carries the previous window's end; first window has `sq->end = 0`):
/// ```c
/// if (W > 0) { /* forward strand */
///   sq->C     = ESL_MIN(sq->n, C);          /* context overlap from prev window */
///   sq->start = sq->end - sq->C + 1;
///   sq->end   = ESL_MIN(tmpsq->L, sq->end + W);
///   sq->n     = sq->end - sq->start + 1;
///   sq->W     = sq->n - sq->C;               /* # of NEW residues */
/// }
/// ```
/// We re-derive the identical window coords from the in-memory full strand instead
/// of a stateful file reader. `full_dsq` is 1-indexed with sentinels (residues
/// `1..=l`); returned windows are in GLOBAL 1-based coordinates.
pub fn f1_filter_sequence(f: &MsvFilter, full_dsq: &[u8], l: usize, f1: f64, maxw: usize, cmw: i64) -> Vec<Window> {
    let mut out: Vec<Window> = Vec::new();
    let mut new_start = 1usize; // global position of this window's first NEW residue
    let mut first = true;
    while new_start <= l {
        // sq->C = MIN(prev_n, maxW) context; sq->start = sq->end - C + 1.
        let ctx = if first { 0 } else { maxw.min(new_start - 1) };
        let win_global_start = new_start - ctx; // 1-based global
        // sq->end = MIN(L, sq->end + W): W new residues (fewer at end of sequence).
        let new_count = CM_MAX_RESIDUE_COUNT.min(l - new_start + 1);
        let win_len = ctx + new_count;
        let win_global_end = win_global_start + win_len - 1;

        // Build the window's own sentinel-padded dsq (local coords 1..=win_len).
        let mut wdsq = Vec::with_capacity(win_len + 2);
        wdsq.push(255u8);
        wdsq.extend_from_slice(&full_dsq[win_global_start..=win_global_end]);
        wdsq.push(255u8);

        for w in f1_msv_windows(f, &wdsq, win_len, f1, cmw) {
            out.push(Window {
                start: win_global_start as i64 + w.start - 1,
                end: win_global_start as i64 + w.end - 1,
                score: w.score,
            });
        }

        new_start = win_global_end + 1;
        first = false;
    }
    out
}

// ===========================================================================
// F3: local Forward filter
// ===========================================================================
//
// Faithful transcription of HMMER's `forward_engine` (impl_sse/fwdback.c:256,
// the `p7_ForwardParser` variant) plus the `p7_ProfileConfig`/`p7_ReconfigLength`
// (modelconfig.c) local-multihit profile configuration and `fb_conversion`
// (impl_sse/p7_oprofile.c:934) odds-ratio conversion that feeds it.
//
// The optimized profile stores, in probability/odds space:
//   - rfv[x][k] = exp(MSC[k][x])          (match-emission odds ratio; ISC = 1)
//   - tfv transitions = the raw HMM transition probabilities exp(log(t))
//   - tBM[k]   = occ[k]/Z                 (occupancy-weighted local entry)
//   - xf[E][*] = 0.5                       (multihit; exp(-log2))
//   - xf[N/C/J][LOOP|MOVE] = ploop|pmove   (length model, set per window)
// The DP runs in odds space with sparse rescaling when xE > 1e4 (matching C),
// and the returned score is `totscale + log(xC * xf[C][MOVE])` nats.

/// Configured local-multihit p7 profile in odds/probability space, for the
/// Forward filter. Length-independent parts; the N/C/J length model is applied
/// per target-window length inside [`forward_filter_score`].
pub struct ForwardFilter {
    pub m: usize,
    /// rfv[k][x] = exp(MSC[k][x]) — match emission odds ratio (x in 0..KP).
    rfv: Vec<[f32; KP]>,
    /// Transposed emission table for SIMD: rfv_t[x*(m+1) + k] = rfv[k][x], so a
    /// fixed residue x has its per-k odds contiguous (vectorizable inner loop).
    rfv_t: Vec<f32>,
    /// Local entry B->M_k = occ[k]/Z, k=1..=M (index 0 unused).
    tbm: Vec<f32>,
    /// Transitions INTO M_k from node k-1: amm=M->M, aim=I->M, adm=D->M. k=1..=M.
    amm: Vec<f32>,
    aim: Vec<f32>,
    adm: Vec<f32>,
    /// Insert transitions at node k: tmi=M_k->I_k, tii=I_k->I_k. 0 at k=M.
    tmi: Vec<f32>,
    tii: Vec<f32>,
    /// Delete transitions out of node k: tmd=M_k->D_{k+1}, tdd=D_k->D_{k+1}. 0 at k=M.
    tmd: Vec<f32>,
    tdd: Vec<f32>,
    pub ftau: f64,
    pub flambda: f64,
    // --- Striped (Farrar) packed profile, for the C-faithful striped Forward
    // engine (fwdback.c forward_engine). nq = p7O_NQF(M) = max(2,(M-1)/4+1) SSE
    // vectors. Cell (q,z) maps to model position p = q+1+z*nq. Packed from the
    // SAME natural-order arrays the scalar path uses (so only float summation
    // ORDER differs, not the operand values). ---
    nq: usize,
    /// Striped main transitions, 7 per vector in C order [BM,MM,IM,DM,MD,MI,II],
    /// laid out tmain[q*(7*4) + t*4 + z]. exp(-inf)=0 padding for p>M.
    tmain: Vec<f32>,
    /// Striped DD transitions tddv[q*4 + z].
    tddv: Vec<f32>,
    /// Striped emission odds rfvv[x*(nq*4) + q*4 + z] = rfv[p][x] (0 for p>M).
    rfvv: Vec<f32>,
}

/// Build the Forward filter profile from the p7 filter HMM. Faithful to
/// `p7_ProfileConfig` (LOCAL multihit) + `fb_conversion`.
pub fn build_forward_filter(p7: &P7Profile) -> ForwardFilter {
    let m = p7.m as usize;

    // Match emission scores MSC[k][x], then rfv = exp(MSC) (fb_conversion). The
    // scores are modelconfig.c:141-151 (bg->f[x] = 1/K = 0.25):
    //   for (k = 1; k <= hmm->M; k++) {
    //     for (x = 0; x < hmm->abc->K; x++)  sc[x] = log((double)hmm->mat[k][x] / bg->f[x]);
    //     esl_abc_FExpectScVec(hmm->abc, sc, bg->f);
    //     for (x = 0; x < hmm->abc->Kp; x++) { rp = gm->rsc[x] + k*p7P_NR; rp[p7P_MSC] = sc[x]; }
    //   }
    // fb_conversion (p7_oprofile.c:960): om->rfv[x][q] = esl_sse_expf(MSC) (we use libm exp).
    let mut rfv = vec![[0.0f32; KP]; m + 1];
    for k in 1..=m {
        let mut sc = [f32::NEG_INFINITY; KP];
        for x in 0..K_CANON {
            sc[x] = ((p7.mat[k][x] as f64) / 0.25).ln() as f32;
        }
        // esl_abc_FExpectScVec: degenerate = mean of canonical scores in the set
        // (uniform bg → simple mean over degen_set(code)).
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
            rfv[k][x] = sc[x].exp(); // exp(-inf) = 0
        }
    }
    // Transposed copy for the SIMD inner loop: rfv_t[x*(m+1)+k] = rfv[k][x].
    let mut rfv_t = vec![0.0f32; KP * (m + 1)];
    for k in 0..=m {
        for x in 0..KP {
            rfv_t[x * (m + 1) + k] = rfv[k][x];
        }
    }

    // Occupancy mocc[k]. C p7_hmm.c::p7_hmm_CalculateOccupancy (trans layout:
    // [MM=0, MI=1, MD=2, IM=3, II=4, DM=5, DD=6]):
    //   mocc[1] = hmm->t[0][p7H_MI] + hmm->t[0][p7H_MM];
    //   for (k = 2; k <= hmm->M; k++)
    //     mocc[k] = mocc[k-1] * (hmm->t[k-1][p7H_MM] + hmm->t[k-1][p7H_MI]) +
    //               (1.0-mocc[k-1]) * hmm->t[k-1][p7H_DM];
    let mut occ = vec![0.0f32; m + 1];
    occ[0] = 0.0;
    if m >= 1 {
        occ[1] = p7.trans[0][1] + p7.trans[0][0];
    }
    for k in 2..=m {
        occ[k] = occ[k - 1] * (p7.trans[k - 1][0] + p7.trans[k - 1][1])
            + (1.0 - occ[k - 1]) * p7.trans[k - 1][5];
    }
    // Local entry. C modelconfig.c:90-97 (IsLocal branch) + fb_conversion exp:
    //   Z = 0.;
    //   for (k = 1; k <= hmm->M; k++)  Z += occ[k] * (float) (hmm->M-k+1);
    //   for (k = 1; k <= hmm->M; k++)  p7P_TSC(gm, k-1, p7P_BM) = log(occ[k] / Z);
    // tBM[k] = exp(log(occ[k]/Z)) — round-trips through log/exp exactly as C does.
    let mut z = 0.0f32;
    for k in 1..=m {
        z += occ[k] * (m - k + 1) as f32;
    }
    let mut tbm = vec![0.0f32; m + 1];
    for k in 1..=m {
        tbm[k] = ((occ[k] / z) as f64).ln().exp() as f32;
    }

    // Transition scores. C modelconfig.c:126-135 sets tsc[k] = log(hmm->t[k][...])
    // for k=1..M-1; fb_conversion exp's them back to the raw probabilities. The
    // transitions INTO M_k use node k-1 (fb_conversion kb=k-1 for BM/MM/IM/DM):
    //   tp[p7P_MM]=log(hmm->t[k][p7H_MM]); tp[p7P_IM]=log(hmm->t[k][p7H_IM]);
    //   tp[p7P_DM]=log(hmm->t[k][p7H_DM]); (also MI,MD,II,DD)
    let mut amm = vec![0.0f32; m + 1];
    let mut aim = vec![0.0f32; m + 1];
    let mut adm = vec![0.0f32; m + 1];
    for k in 1..=m {
        amm[k] = p7.trans[k - 1][0]; // M_{k-1}->M_k
        aim[k] = p7.trans[k - 1][3]; // I_{k-1}->M_k
        adm[k] = p7.trans[k - 1][5]; // D_{k-1}->M_k
    }
    // Insert / delete-out transitions at node k; impossible (0) at k=M because
    // fb_conversion's `kb+z*nq < M` test is false there (kb=k=M).
    let mut tmi = vec![0.0f32; m + 1];
    let mut tii = vec![0.0f32; m + 1];
    let mut tmd = vec![0.0f32; m + 1];
    let mut tdd = vec![0.0f32; m + 1];
    for k in 1..m {
        tmi[k] = p7.trans[k][1]; // M_k->I_k
        tii[k] = p7.trans[k][4]; // I_k->I_k
        tmd[k] = p7.trans[k][2]; // M_k->D_{k+1}
        tdd[k] = p7.trans[k][6]; // D_k->D_{k+1}
    }

    // --- Striped (Farrar) packing, mirroring fb_conversion (p7_oprofile.c:939).
    // nq = p7O_NQF(M) = max(2,(M-1)/4+1); cell (q,z) -> model position
    // p = q+1+z*nq. All values pulled from the natural arrays above (index p when
    // p<=m, else 0), so the striped path uses identical operand values to the
    // scalar path — only the SIMD summation order differs.
    let nq = if m >= 1 { std::cmp::max(2, (m - 1) / 4 + 1) } else { 2 };
    let mut tmain = vec![0.0f32; nq * 7 * 4];
    let mut tddv = vec![0.0f32; nq * 4];
    let mut rfvv = vec![0.0f32; KP * nq * 4];
    let at = |arr: &Vec<f32>, p: usize| -> f32 { if p <= m { arr[p] } else { 0.0 } };
    for q in 0..nq {
        for z in 0..4 {
            let p = q + 1 + z * nq; // model position for this lane
            // C order: [BM, MM, IM, DM, MD, MI, II]. exp(-inf)=0 when p>M.
            tmain[q * 28 + 0 * 4 + z] = at(&tbm, p);
            tmain[q * 28 + 1 * 4 + z] = at(&amm, p);
            tmain[q * 28 + 2 * 4 + z] = at(&aim, p);
            tmain[q * 28 + 3 * 4 + z] = at(&adm, p);
            tmain[q * 28 + 4 * 4 + z] = at(&tmd, p);
            tmain[q * 28 + 5 * 4 + z] = at(&tmi, p);
            tmain[q * 28 + 6 * 4 + z] = at(&tii, p);
            tddv[q * 4 + z] = at(&tdd, p);
            for x in 0..KP {
                rfvv[x * (nq * 4) + q * 4 + z] = if p <= m { rfv[p][x] } else { 0.0 };
            }
        }
    }

    ForwardFilter {
        m,
        rfv,
        rfv_t,
        tbm,
        amm,
        aim,
        adm,
        tmi,
        tii,
        tmd,
        tdd,
        ftau: p7.evparam.lftau,
        flambda: p7.evparam.lflambda,
        nq,
        tmain,
        tddv,
        rfvv,
    }
}

/// `p7_bg_NullOne` (p7_bg.c:357), with `p7_bg_SetLength(bg, L)` setting p1 = L/(L+1):
/// ```c
/// int p7_bg_NullOne(const P7_BG *bg, const ESL_DSQ *dsq, int L, float *ret_sc) {
///   *ret_sc = (float) L * log(bg->p1) + log(1.-bg->p1);   /* bg->p1 = (float)L/(float)(L+1) */
///   return eslOK;
/// }
/// ```
pub fn p7_bg_null_one(l: usize) -> f64 {
    let p1 = l as f64 / (l as f64 + 1.0);
    l as f64 * p1.ln() + (1.0 - p1).ln()
}

/// Scalar transcription of `forward_engine(do_full=FALSE, ...)` (the
/// `p7_ForwardParser`): local-multihit Forward in odds space with sparse
/// rescaling. `dsq` is 1-indexed with sentinels; window length `l`. Returns the
/// Forward score in nats. The length model (xf[N/C/J]) is configured for `l`.
thread_local! {
    // Per-thread reusable DP rows for forward_filter_score (avoids 6 heap allocs
    // per window call). [mp, ip, dp, mc, ic, dc].
    static FF_SCRATCH: std::cell::RefCell<[Vec<f32>; 6]> =
        const { std::cell::RefCell::new([Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()]) };
}

// M/I row update for one residue. Each mc[k]/ic[k] depends only on the previous
// row (mp/ip/dp) and the scalar xb, so lanes are independent and the per-element
// arithmetic (and its op order) is identical whether done 8-wide, 4-wide, or
// scalar. Widening SSE→AVX2 is therefore bit-for-bit identical, not merely close.
//
// AVX2 path: 8-wide, then a 4-wide SSE step, then scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ff_mi_row_avx2(
    ff: &ForwardFilter, m: usize, xb: f32,
    mp: &[f32], ip: &[f32], dp: &[f32], mc: &mut [f32], ic: &mut [f32], rfvx: &[f32],
) {
    use std::arch::x86_64::*;
    let xbv = _mm256_set1_ps(xb);
    let mut k = 1usize;
    while k + 8 <= m + 1 {
        let mut sv = _mm256_mul_ps(xbv, _mm256_loadu_ps(ff.tbm.as_ptr().add(k)));
        sv = _mm256_add_ps(sv, _mm256_mul_ps(_mm256_loadu_ps(mp.as_ptr().add(k - 1)), _mm256_loadu_ps(ff.amm.as_ptr().add(k))));
        sv = _mm256_add_ps(sv, _mm256_mul_ps(_mm256_loadu_ps(ip.as_ptr().add(k - 1)), _mm256_loadu_ps(ff.aim.as_ptr().add(k))));
        sv = _mm256_add_ps(sv, _mm256_mul_ps(_mm256_loadu_ps(dp.as_ptr().add(k - 1)), _mm256_loadu_ps(ff.adm.as_ptr().add(k))));
        _mm256_storeu_ps(mc.as_mut_ptr().add(k), _mm256_mul_ps(sv, _mm256_loadu_ps(rfvx.as_ptr().add(k))));
        let iv = _mm256_add_ps(
            _mm256_mul_ps(_mm256_loadu_ps(mp.as_ptr().add(k)), _mm256_loadu_ps(ff.tmi.as_ptr().add(k))),
            _mm256_mul_ps(_mm256_loadu_ps(ip.as_ptr().add(k)), _mm256_loadu_ps(ff.tii.as_ptr().add(k))),
        );
        _mm256_storeu_ps(ic.as_mut_ptr().add(k), iv);
        k += 8;
    }
    let xbv4 = _mm_set1_ps(xb);
    while k + 4 <= m + 1 {
        let mut sv = _mm_mul_ps(xbv4, _mm_loadu_ps(ff.tbm.as_ptr().add(k)));
        sv = _mm_add_ps(sv, _mm_mul_ps(_mm_loadu_ps(mp.as_ptr().add(k - 1)), _mm_loadu_ps(ff.amm.as_ptr().add(k))));
        sv = _mm_add_ps(sv, _mm_mul_ps(_mm_loadu_ps(ip.as_ptr().add(k - 1)), _mm_loadu_ps(ff.aim.as_ptr().add(k))));
        sv = _mm_add_ps(sv, _mm_mul_ps(_mm_loadu_ps(dp.as_ptr().add(k - 1)), _mm_loadu_ps(ff.adm.as_ptr().add(k))));
        _mm_storeu_ps(mc.as_mut_ptr().add(k), _mm_mul_ps(sv, _mm_loadu_ps(rfvx.as_ptr().add(k))));
        _mm_storeu_ps(ic.as_mut_ptr().add(k),
            _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(mp.as_ptr().add(k)), _mm_loadu_ps(ff.tmi.as_ptr().add(k))),
                       _mm_mul_ps(_mm_loadu_ps(ip.as_ptr().add(k)), _mm_loadu_ps(ff.tii.as_ptr().add(k)))));
        k += 4;
    }
    while k <= m {
        let sv = xb * ff.tbm[k] + mp[k - 1] * ff.amm[k] + ip[k - 1] * ff.aim[k] + dp[k - 1] * ff.adm[k];
        mc[k] = sv * rfvx[k];
        ic[k] = mp[k] * ff.tmi[k] + ip[k] * ff.tii[k];
        k += 1;
    }
}

// Relaxed DD recurrence using FMA: dc[k] = fma(dc[k-1], tdd[k-1], mc[k-1]*tmd[k-1]).
// Breaks strict byte-parity with C (one rounding instead of two on that add) but
// shortens the serial loop-carried critical path from ~(mul+add) to ~(fma)
// latency — the mc*tmd product is off the carried path. Enabled only when the
// biological hit set is verified unchanged (bio_verify harness).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn ff_dd_fma(m: usize, mc: &[f32], dc: &mut [f32], tmd: &[f32], tdd: &[f32]) {
    for k in 2..=m {
        *dc.get_unchecked_mut(k) = f32::mul_add(
            *dc.get_unchecked(k - 1),
            *tdd.get_unchecked(k - 1),
            *mc.get_unchecked(k - 1) * *tmd.get_unchecked(k - 1),
        );
    }
}

// Strict byte-parity toggle for the Forward filter. When true, the relaxations
// that break bit-identity with C (FMA-fused DD recurrence, reordered AVX2 xE sum)
// are disabled; the bit-identical AVX2 M/I widening stays on. Set once at startup
// from the `--strict` CLI flag; read per call (Relaxed is fine — no ordering deps).
static FORWARD_STRICT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Force the Forward filter into strict byte-parity mode (disables FMA/reorder
/// relaxations). Call before running the pipeline.
pub fn set_forward_strict(v: bool) {
    FORWARD_STRICT.store(v, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn forward_strict() -> bool {
    FORWARD_STRICT.load(std::sync::atomic::Ordering::Relaxed)
}

// Sum a[from..=to] with 8-wide AVX2 partial accumulators + horizontal sum.
// Reorders the additions vs a strict left-to-right scalar sum, so the result
// differs by a few ULP (breaks byte-parity) — used only where the value feeds
// the forward FILTER gate (bio-equivalence verified), not a reported score.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_slice_avx2(a: &[f32], from: usize, to_incl: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut k = from;
    while k + 8 <= to_incl + 1 {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a.as_ptr().add(k)));
        k += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    while k <= to_incl {
        s += *a.get_unchecked(k);
        k += 1;
    }
    s
}

// Toggle for the C-faithful striped (Farrar) Forward engine. ON by default (it is
// ~18-32% faster on HMM-filter-dominated small/mid models and verified
// tblout-byte-identical to C across a broad battery). Set INFERNOX_FF_STRIPED=0 to
// fall back to the scalar/AVX2 reference path (used for A/B comparison and as a
// trivial revert). The striped path is also skipped under `--strict` (which wants
// the scalar reference), and on non-SSE2 hardware.
#[cfg(target_arch = "x86_64")]
fn forward_striped_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("INFERNOX_FF_STRIPED").as_deref() != Ok("0"))
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    // Striped single-row scratch (do_full=FALSE): [mm, dm, im], each 4*nq floats.
    static FF_STRIPED_SCRATCH: std::cell::RefCell<[Vec<f32>; 3]> =
        const { std::cell::RefCell::new([Vec::new(), Vec::new(), Vec::new()]) };
}

/// Faithful transcription of `forward_engine(do_full=FALSE)` (fwdback.c:255-463)
/// in the striped (Farrar) SSE layout: the exact 4-wide op order, the
/// rightshift-zero carry across stripe boundaries, the M<100 fully-serialized DD
/// vs the M>=100 lazy-convergence-break DD (fwdback.c:366-397). Cell (q,z) maps to
/// model position p = q+1+z*nq. Uses the striped packed profile (tmain/tddv/rfvv).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn forward_filter_score_striped(ff: &ForwardFilter, dsq: &[u8], l: usize) -> f32 {
    use std::arch::x86_64::*;
    let m = ff.m;
    let q = ff.nq;

    // Length model (identical to the scalar path).
    let nj = 1.0f32;
    let pmove = (2.0 + nj) / (l as f32 + 2.0 + nj);
    let ploop = 1.0 - pmove;
    let (xf_e_move, xf_e_loop) = (0.5f32, 0.5f32);
    let (xf_n_loop, xf_n_move) = (ploop, pmove);
    let (xf_c_loop, xf_c_move) = (ploop, pmove);
    let (xf_j_loop, xf_j_move) = (ploop, pmove);

    let (mut mm, mut dm, mut im) = FF_STRIPED_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        (std::mem::take(&mut s[0]), std::mem::take(&mut s[1]), std::mem::take(&mut s[2]))
    });
    for b in [&mut mm, &mut dm, &mut im] {
        b.clear();
        b.resize(4 * q, 0.0);
    }
    let mmp = mm.as_mut_ptr();
    let dmp = dm.as_mut_ptr();
    let imp = im.as_mut_ptr();
    let tp0 = ff.tmain.as_ptr();
    let tdd0 = ff.tddv.as_ptr();

    #[inline(always)]
    unsafe fn rsz(a: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
        use std::arch::x86_64::*;
        _mm_castsi128_ps(_mm_slli_si128::<4>(_mm_castps_si128(a)))
    }

    let zerov = _mm_setzero_ps();
    let mut xn = 1.0f32;
    let mut xj = 0.0f32;
    let mut xb = xf_n_move;
    let mut xc = 0.0f32;
    let mut totscale = 0.0f32;

    for i in 1..=l {
        let x = dsq[i] as usize;
        let rp0 = ff.rfvv.as_ptr().add(x * (q * 4));
        let mut dcv = zerov;
        let mut xev = zerov;
        let xbv = _mm_set1_ps(xb);
        // Right-shifts of the last vector (prev row) — fwdback.c:305-307.
        let mut mpv = rsz(_mm_loadu_ps(mmp.add((q - 1) * 4)));
        let mut dpv = rsz(_mm_loadu_ps(dmp.add((q - 1) * 4)));
        let mut ipv = rsz(_mm_loadu_ps(imp.add((q - 1) * 4)));

        // Main q-loop (fwdback.c:309-338).
        for qi in 0..q {
            let tb = qi * 28;
            let mut sv = _mm_mul_ps(xbv, _mm_loadu_ps(tp0.add(tb)));
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv, _mm_loadu_ps(tp0.add(tb + 4))));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv, _mm_loadu_ps(tp0.add(tb + 8))));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv, _mm_loadu_ps(tp0.add(tb + 12))));
            sv = _mm_mul_ps(sv, _mm_loadu_ps(rp0.add(qi * 4)));
            xev = _mm_add_ps(xev, sv);
            // Reload prev-row {M,D,I}(q) BEFORE overwriting (delayed store).
            mpv = _mm_loadu_ps(mmp.add(qi * 4));
            dpv = _mm_loadu_ps(dmp.add(qi * 4));
            ipv = _mm_loadu_ps(imp.add(qi * 4));
            _mm_storeu_ps(mmp.add(qi * 4), sv);
            _mm_storeu_ps(dmp.add(qi * 4), dcv);
            dcv = _mm_mul_ps(sv, _mm_loadu_ps(tp0.add(tb + 16))); // M->D partial
            let svi = _mm_mul_ps(mpv, _mm_loadu_ps(tp0.add(tb + 20))); // M->I
            _mm_storeu_ps(imp.add(qi * 4),
                _mm_add_ps(svi, _mm_mul_ps(ipv, _mm_loadu_ps(tp0.add(tb + 24))))); // I->I
        }

        // DD paths — first obligatory pass (fwdback.c:349-356).
        dcv = rsz(dcv);
        _mm_storeu_ps(dmp.add(0), zerov);
        for qi in 0..q {
            let d = _mm_add_ps(dcv, _mm_loadu_ps(dmp.add(qi * 4)));
            _mm_storeu_ps(dmp.add(qi * 4), d);
            dcv = _mm_mul_ps(d, _mm_loadu_ps(tdd0.add(qi * 4)));
        }
        if m < 100 {
            // Fully serialized: 3 more passes (fwdback.c:366-378).
            for _j in 1..4 {
                dcv = rsz(dcv);
                for qi in 0..q {
                    let d = _mm_add_ps(dcv, _mm_loadu_ps(dmp.add(qi * 4)));
                    _mm_storeu_ps(dmp.add(qi * 4), d);
                    dcv = _mm_mul_ps(dcv, _mm_loadu_ps(tdd0.add(qi * 4)));
                }
            }
        } else {
            // Lazy convergence break (fwdback.c:379-397).
            for _j in 1..4 {
                let mut cv = zerov;
                dcv = rsz(dcv);
                for qi in 0..q {
                    let old = _mm_loadu_ps(dmp.add(qi * 4));
                    let d = _mm_add_ps(dcv, old);
                    cv = _mm_or_ps(cv, _mm_cmpgt_ps(d, old));
                    _mm_storeu_ps(dmp.add(qi * 4), d);
                    dcv = _mm_mul_ps(dcv, _mm_loadu_ps(tdd0.add(qi * 4)));
                }
                if _mm_movemask_ps(cv) == 0 {
                    break;
                }
            }
        }

        // Add D's to xE (fwdback.c:400), then horizontal sum (fwdback.c:407-409).
        for qi in 0..q {
            xev = _mm_add_ps(xev, _mm_loadu_ps(dmp.add(qi * 4)));
        }
        xev = _mm_add_ps(xev, _mm_shuffle_ps::<0x39>(xev, xev)); // _MM_SHUFFLE(0,3,2,1)
        xev = _mm_add_ps(xev, _mm_shuffle_ps::<0x4E>(xev, xev)); // _MM_SHUFFLE(1,0,3,2)
        let xe = _mm_cvtss_f32(xev);

        // Specials (fwdback.c:411-414).
        xn = xn * xf_n_loop;
        xc = (xc * xf_c_loop) + (xe * xf_e_move);
        xj = (xj * xf_j_loop) + (xe * xf_e_loop);
        xb = (xj * xf_j_move) + (xn * xf_n_move);

        // Sparse rescaling (fwdback.c:417-434).
        if xe > 1.0e4 {
            let inv = 1.0 / xe;
            xn *= inv;
            xc *= inv;
            xj *= inv;
            xb *= inv;
            let invv = _mm_set1_ps(inv);
            for qi in 0..q {
                _mm_storeu_ps(mmp.add(qi * 4), _mm_mul_ps(_mm_loadu_ps(mmp.add(qi * 4)), invv));
                _mm_storeu_ps(dmp.add(qi * 4), _mm_mul_ps(_mm_loadu_ps(dmp.add(qi * 4)), invv));
                _mm_storeu_ps(imp.add(qi * 4), _mm_mul_ps(_mm_loadu_ps(imp.add(qi * 4)), invv));
            }
            totscale += (xe as f64).ln() as f32;
        }
    }

    let sc = totscale as f64 + ((xc as f64) * (xf_c_move as f64)).ln();
    FF_STRIPED_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        s[0] = mm;
        s[1] = dm;
        s[2] = im;
    });
    sc as f32
}

pub fn forward_filter_score(ff: &ForwardFilter, dsq: &[u8], l: usize) -> f32 {
    let m = ff.m;
    #[cfg(target_arch = "x86_64")]
    let strict = forward_strict();
    #[cfg(target_arch = "x86_64")]
    {
        // C-faithful striped (Farrar) engine: default fast path. Skipped under
        // --strict (scalar reference) or INFERNOX_FF_STRIPED=0.
        if !strict && forward_striped_enabled() && is_x86_feature_detected!("sse2") {
            return unsafe { forward_filter_score_striped(ff, dsq, l) };
        }
    }
    // M/I widening is bit-identical, so it runs regardless of `strict`.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2");
    // DD-FMA and the reordered xE sum break byte-parity → gated off by `strict`.
    #[cfg(target_arch = "x86_64")]
    let use_fma = is_x86_feature_detected!("fma") && !strict;
    #[cfg(target_arch = "x86_64")]
    let use_xe_avx2 = use_avx2 && !strict;

    // Length model. C p7_ReconfigLength (modelconfig.c:228-231; nj=1 for multihit),
    // then fb_conversion exp of xsc gives the odds xf:
    //   pmove = (2.0f + gm->nj) / ((float) L + 2.0f + gm->nj);   /* 2/(L+2) sw; 3/(L+3) fs */
    //   ploop = 1.0f - pmove;
    //   gm->xsc[N|C|J][LOOP] = log(ploop);  gm->xsc[N|C|J][MOVE] = log(pmove);
    // xf[E][LOOP]=xf[E][MOVE]=exp(-eslCONST_LOG2)=0.5 (multihit, modelconfig.c:116-117).
    let nj = 1.0f32;
    let pmove = (2.0 + nj) / (l as f32 + 2.0 + nj);
    let ploop = 1.0 - pmove;
    let xf_e_move = 0.5f32; // exp(-log2)
    let xf_e_loop = 0.5f32;
    let xf_n_loop = ploop;
    let xf_n_move = pmove;
    let xf_c_loop = ploop;
    let xf_c_move = pmove;
    let xf_j_loop = ploop;
    let xf_j_move = pmove;

    // Current/previous DP rows in odds space (k = 0..=M; index 0 is the
    // right-shifted zero from striped SIMD, always 0). Pulled from a per-thread
    // scratch pool and re-zeroed (removes 6 heap allocs per window call).
    let (mut mp, mut ip, mut dp, mut mc, mut ic, mut dc) = FF_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        (
            std::mem::take(&mut s[0]), std::mem::take(&mut s[1]), std::mem::take(&mut s[2]),
            std::mem::take(&mut s[3]), std::mem::take(&mut s[4]), std::mem::take(&mut s[5]),
        )
    });
    for b in [&mut mp, &mut ip, &mut dp, &mut mc, &mut ic, &mut dc] {
        b.clear();
        b.resize(m + 1, 0.0);
    }

    // Specials (forward_engine init).
    let mut xe;
    let mut xn = 1.0f32;
    let mut xj = 0.0f32;
    let mut xb = xf_n_move;
    let mut xc = 0.0f32;
    let mut totscale = 0.0f32;

    for i in 1..=l {
        let x = dsq[i] as usize;
        mc[0] = 0.0;
        ic[0] = 0.0;
        dc[0] = 0.0;
        // M and I from the previous row. C fwdback.c:312-337 (striped SIMD; here
        // the equivalent unstriped scalar, so mp[k-1] plays the role of the
        // right-shifted `mpv`/`ipv`/`dpv` and `*tp` steps are our per-node arrays):
        //   sv   =                _mm_mul_ps(xBv, *tp);  tp++;   // B->Mk  (tbm)
        //   sv   = _mm_add_ps(sv, _mm_mul_ps(mpv, *tp)); tp++;   // Mk-1->Mk (amm)
        //   sv   = _mm_add_ps(sv, _mm_mul_ps(ipv, *tp)); tp++;   // Ik-1->Mk (aim)
        //   sv   = _mm_add_ps(sv, _mm_mul_ps(dpv, *tp)); tp++;   // Dk-1->Mk (adm)
        //   sv   = _mm_mul_ps(sv, *rp);                  rp++;   // * rfv emission
        //   ...
        //   sv         =                _mm_mul_ps(mpv, *tp);  tp++;  // Mk->Ik (tmi)
        //   IMO(dpc,q) = _mm_add_ps(sv, _mm_mul_ps(ipv, *tp)); tp++;  // Ik->Ik (tii); ins emission odds = 1
        // M and I from the previous row. Each mc[k]/ic[k] is INDEPENDENT (reads
        // only the previous row + scalar xb), so 4-wide SSE is bit-identical to
        // the scalar per-element math (same op order per lane). rfv_t gives the
        // emission odds contiguously for residue x. dc + xE stay scalar below.
        let rfvx = &ff.rfv_t[x * (m + 1)..];
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            unsafe { ff_mi_row_avx2(ff, m, xb, &mp, &ip, &dp, &mut mc, &mut ic, rfvx) };
        } else {
            unsafe {
            use std::arch::x86_64::*;
            let xbv = _mm_set1_ps(xb);
            let mut k = 1usize;
            while k + 4 <= m + 1 {
                let tbmv = _mm_loadu_ps(ff.tbm.as_ptr().add(k));
                let mpm1 = _mm_loadu_ps(mp.as_ptr().add(k - 1));
                let ammv = _mm_loadu_ps(ff.amm.as_ptr().add(k));
                let ipm1 = _mm_loadu_ps(ip.as_ptr().add(k - 1));
                let aimv = _mm_loadu_ps(ff.aim.as_ptr().add(k));
                let dpm1 = _mm_loadu_ps(dp.as_ptr().add(k - 1));
                let admv = _mm_loadu_ps(ff.adm.as_ptr().add(k));
                let mut sv = _mm_mul_ps(xbv, tbmv);
                sv = _mm_add_ps(sv, _mm_mul_ps(mpm1, ammv));
                sv = _mm_add_ps(sv, _mm_mul_ps(ipm1, aimv));
                sv = _mm_add_ps(sv, _mm_mul_ps(dpm1, admv));
                let rfvv = _mm_loadu_ps(rfvx.as_ptr().add(k));
                _mm_storeu_ps(mc.as_mut_ptr().add(k), _mm_mul_ps(sv, rfvv));

                let mpk = _mm_loadu_ps(mp.as_ptr().add(k));
                let tmiv = _mm_loadu_ps(ff.tmi.as_ptr().add(k));
                let ipk = _mm_loadu_ps(ip.as_ptr().add(k));
                let tiiv = _mm_loadu_ps(ff.tii.as_ptr().add(k));
                _mm_storeu_ps(ic.as_mut_ptr().add(k),
                    _mm_add_ps(_mm_mul_ps(mpk, tmiv), _mm_mul_ps(ipk, tiiv)));
                k += 4;
            }
            while k <= m {
                let sv = xb * ff.tbm[k] + mp[k - 1] * ff.amm[k]
                    + ip[k - 1] * ff.aim[k] + dp[k - 1] * ff.adm[k];
                mc[k] = sv * rfvx[k];
                ic[k] = mp[k] * ff.tmi[k] + ip[k] * ff.tii[k];
                k += 1;
            }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for k in 1..=m {
            let sv = xb * ff.tbm[k] + mp[k - 1] * ff.amm[k]
                + ip[k - 1] * ff.aim[k] + dp[k - 1] * ff.adm[k];
            mc[k] = sv * rfvx[k];
            ic[k] = mp[k] * ff.tmi[k] + ip[k] * ff.tii[k];
        }
        // The DD paths. C fwdback.c:349-397 does 4 striped passes (full
        // serialization for M<100); the scalar left-to-right sweep is exactly
        // that fully-serialized result:
        //   DMO(dpc,q) = _mm_add_ps(dcv, DMO(dpc,q));      // += M->D and D->D
        //   dcv        = _mm_mul_ps(DMO(dpc,q), *tp);      // extend (tmd / tdd)
        dc[1] = 0.0;
        // Serial recurrence (dc[k] needs dc[k-1]). Byte-parity path is 2 roundings
        // (mul + add); the FMA path fuses to 1 rounding and a shorter critical path.
        #[cfg(target_arch = "x86_64")]
        {
            if use_fma {
                unsafe { ff_dd_fma(m, &mc, &mut dc, &ff.tmd, &ff.tdd) };
            } else {
                for k in 2..=m {
                    unsafe {
                        *dc.get_unchecked_mut(k) = *mc.get_unchecked(k - 1) * *ff.tmd.get_unchecked(k - 1)
                            + *dc.get_unchecked(k - 1) * *ff.tdd.get_unchecked(k - 1);
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for k in 2..=m {
            unsafe {
                *dc.get_unchecked_mut(k) = *mc.get_unchecked(k - 1) * *ff.tmd.get_unchecked(k - 1)
                    + *dc.get_unchecked(k - 1) * *ff.tdd.get_unchecked(k - 1);
            }
        }
        // xE = sum_k M(i,k) + sum_k D(i,k). C fwdback.c:317 (xEv += sv per M) and
        // :400 (`for (q...) xEv = _mm_add_ps(DMO(dpc,q), xEv);`), then horiz-sum.
        // Keep the summation order (reordering would change the float result);
        // only the bounds checks are elided.
        #[cfg(target_arch = "x86_64")]
        {
            if use_xe_avx2 {
                xe = unsafe { sum_slice_avx2(&mc, 1, m) + sum_slice_avx2(&dc, 1, m) };
            } else {
                let mut xev = 0.0f32;
                for k in 1..=m {
                    xev += unsafe { *mc.get_unchecked(k) };
                }
                for k in 1..=m {
                    xev += unsafe { *dc.get_unchecked(k) };
                }
                xe = xev;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut xev = 0.0f32;
            for k in 1..=m {
                xev += unsafe { *mc.get_unchecked(k) };
            }
            for k in 1..=m {
                xev += unsafe { *dc.get_unchecked(k) };
            }
            xe = xev;
        }

        // Specials. C fwdback.c:411-414 (exact ordering: xN then xC then xJ then
        // xB, so xB sees the NEW xN/xJ):
        //   xN =  xN * om->xf[p7O_N][p7O_LOOP];
        //   xC = (xC * om->xf[p7O_C][p7O_LOOP]) +  (xE * om->xf[p7O_E][p7O_MOVE]);
        //   xJ = (xJ * om->xf[p7O_J][p7O_LOOP]) +  (xE * om->xf[p7O_E][p7O_LOOP]);
        //   xB = (xJ * om->xf[p7O_J][p7O_MOVE]) +  (xN * om->xf[p7O_N][p7O_MOVE]);
        xn = xn * xf_n_loop;
        xc = (xc * xf_c_loop) + (xe * xf_e_move);
        xj = (xj * xf_j_loop) + (xe * xf_e_loop);
        xb = (xj * xf_j_move) + (xn * xf_n_move);

        // Sparse rescaling. C fwdback.c:417-435:
        //   if (xE > 1.0e4) { xN/=xE; xC/=xE; xJ/=xE; xB/=xE;
        //                     xEv=_mm_set1_ps(1.0/xE); for(q) {MMO,DMO,IMO} *= xEv;
        //                     ox->totscale += log(xE); xE = 1.0; }
        if xe > 1.0e4 {
            let inv = 1.0 / xe;
            xn *= inv;
            xc *= inv;
            xj *= inv;
            xb *= inv;
            for k in 0..=m {
                unsafe {
                    *mc.get_unchecked_mut(k) *= inv;
                    *dc.get_unchecked_mut(k) *= inv;
                    *ic.get_unchecked_mut(k) *= inv;
                }
            }
            totscale += (xe as f64).ln() as f32;
        }

        // Swap rows.
        std::mem::swap(&mut mp, &mut mc);
        std::mem::swap(&mut ip, &mut ic);
        std::mem::swap(&mut dp, &mut dc);
    }

    // C->T, and flip the total score back to log space (nats). C fwdback.c:461:
    //   *opt_sc = ox->totscale + log(xC * om->xf[p7O_C][p7O_MOVE]);
    let sc = totscale as f64 + ((xc as f64) * (xf_c_move as f64)).ln();

    // Return the scratch rows to the per-thread pool for the next call.
    FF_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        s[0] = mp; s[1] = ip; s[2] = dp; s[3] = mc; s[4] = ic; s[5] = dc;
    });

    sc as f32
}

// ===========================================================================
// F3b: composition bias filter
// ===========================================================================
//
// Faithful transcription of `p7_bg_FilterScore` (p7_bg.c:471): the score of the
// target under a 2-state "filter null" HMM (`p7_bg_SetFilter` p7_bg.c:429 +
// `esl_hmm_Configure` esl_hmm.c:118 + `esl_hmm_Forward` esl_hmm.c:353), plus the
// same geometric length term as null1. State 0 is the iid background (emission
// odds 1.0 everywhere); state 1 is the model's mean composition `compo`.

/// The 2-state filter-null HMM. Length-independent parts precomputed; the state-0
/// self/switch transitions depend on the target window length (set per call).
pub struct BiasFilter {
    /// eo[code][state] emission odds ratio (esl_hmm eo), state in {0,1}.
    eo: Vec<[f32; 2]>,
    /// State-1 transitions [->0, ->1, ->End]; length-independent (p7_bg_SetFilter).
    t1: [f32; 3],
    /// Initial distribution pi[0], pi[1] (p7_bg_SetFilter: 0.999, 0.001).
    pi: [f32; 2],
}

/// Build the filter HMM from the model composition (`p7_bg_SetFilter` +
/// `esl_hmm_Configure`). `compo` = mean model composition (probabilities), `m` =
/// p7 model length in nodes (for L1 = M/8). Background is uniform 0.25 (RNA).
pub fn build_bias_filter(compo: &[f32; 4], m: usize) -> BiasFilter {
    let bg = 0.25f32;
    // p7_bg_SetFilter (p7_bg.c:429): L1 = (float) M / 8.0; state 0 emits bg->f,
    // state 1 emits `compo`; pi[0]=0.999, pi[1]=0.001.
    let l1 = m as f32 / 8.0;

    // Emission odds eo[x][k] = e[k][x]/fq[x], fq = bg->f. C esl_hmm_Configure
    // (esl_hmm.c:127-153). State 0 (e[0]=bg) → eo[*][0]=1.0. Degenerate codes:
    //   for x in K+1..Kp-3: eo[x][k] = (sum_{y in degen[x]} e[k][y]) / (sum fq[y]);
    //   gap(K)/nonres(Kp-2)/missing(Kp-1): eo = 1.0.
    let mut eo = vec![[1.0f32; 2]; KP];
    for code in 0..KP {
        eo[code][0] = 1.0; // e[0][x] = bg[x] → e[0][x]/bg[x] = 1.0
        let set = degen_set(code);
        if set.is_empty() {
            eo[code][1] = 1.0; // gap / nonresidue / missing (esl_hmm.c:135-137)
        } else {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for &y in set {
                num += compo[y]; // hmm->e[1][y] = compo[y]
                den += bg; // fq[y] = bg
            }
            eo[code][1] = if den > 0.0 { num / den } else { 0.0 };
        }
    }

    // State-1 transitions. C p7_bg_SetFilter (p7_bg.c:441-443):
    //   bg->fhmm->t[1][0] = 1.0f / (L1+1.0f);
    //   bg->fhmm->t[1][1] =   L1 / (L1+1.0f);
    //   bg->fhmm->t[1][2] = 1.0f;   // 1.0 transition to E
    let t1 = [1.0 / (l1 + 1.0), l1 / (l1 + 1.0), 1.0];

    BiasFilter { eo, t1, pi: [0.999, 0.001] }
}

/// `p7_bg_FilterScore` (p7_bg.c:471): filter-null Forward score for the window,
/// in nats. `dsq` 1-indexed with sentinels; window length `l`. Length model set
/// for `l` (p7_bg_SetLength → t[0] and the geometric term).
pub fn bias_filter_score(bf: &BiasFilter, dsq: &[u8], l: usize) -> f32 {
    // p7_bg_SetLength (p7_bg.c): p1 = L/(L+1); fhmm->t[0][0]=p1; fhmm->t[0][1]=1-p1.
    // t[0][2] (to End) stays 1.0 from p7_bg_SetFilter.
    let p1 = l as f32 / (l as f32 + 1.0);
    let t: [[f32; 3]; 2] = [[p1, 1.0 - p1, 1.0], bf.t1];

    // esl_hmm_Forward (esl_hmm.c:352, M=2). Per-row max-rescaling; logsc = sum of
    // the per-row log(max) plus the final transition-to-End term.
    let mut logsc = 0.0f32;
    if l == 0 {
        return 0.0; // esl_hmm.c:362-366 L==0 path (log(pi[M])); unused in pipeline.
    }
    // esl_hmm.c:368-376:
    //   for (k=0;k<M;k++) { fwd->dp[1][k] = hmm->eo[dsq[1]][k]*hmm->pi[k]; max=MAX(...); }
    //   for (k=0;k<M;k++)   fwd->dp[1][k] /= max;
    //   fwd->sc[1] = log(max);
    let x1 = dsq[1] as usize;
    let mut dp = [bf.eo[x1][0] * bf.pi[0], bf.eo[x1][1] * bf.pi[1]];
    let mut max = dp[0].max(dp[1]);
    dp[0] /= max;
    dp[1] /= max;
    logsc += (max as f64).ln() as f32;

    // esl_hmm.c:378-395:
    //   for (i=2;i<=L;i++) { for (k=0;k<M;k++) { fwd->dp[i][k]=0;
    //       for (m=0;m<M;m++) fwd->dp[i][k] += fwd->dp[i-1][m]*hmm->t[m][k];
    //       fwd->dp[i][k] *= hmm->eo[dsq[i]][k]; max=MAX(...); }
    //     for (k=0;k<M;k++) fwd->dp[i][k] /= max;  fwd->sc[i] = log(max); }
    for i in 2..=l {
        let x = dsq[i] as usize;
        let mut nd = [0.0f32; 2];
        for k in 0..2 {
            let mut s = 0.0f32;
            for m in 0..2 {
                s += dp[m] * t[m][k];
            }
            nd[k] = s * bf.eo[x][k];
        }
        max = nd[0].max(nd[1]);
        dp[0] = nd[0] / max;
        dp[1] = nd[1] / max;
        logsc += (max as f64).ln() as f32;
    }

    // esl_hmm.c:398-405: fwd->sc[L+1] = log(sum_m dp[L][m]*t[m][M]); t[m][M=2]=1.0.
    //   logsc = sum_{i=1}^{L+1} fwd->sc[i]   (accumulated above + this term)
    let end = dp[0] * t[0][2] + dp[1] * t[1][2];
    logsc += (end as f64).ln() as f32;

    // p7_bg_FilterScore (p7_bg.c:479) imposes the null1 geometric length term:
    //   *ret_sc = nullsc + (float) L * logf(bg->p1) + logf(1.-bg->p1);
    logsc + (l as f32) * p1.ln() + (1.0f32 - p1).ln()
}

/// Faithful transcription of `pli_p7_filter` (cm_pipeline.c:2481-2895) for the
/// standard cmsearch pass on a <20Mb DB, where Viterbi/F1b/F2b are OFF and
/// F3 (local Forward) + F3b (Forward composition bias) are ON.
///
/// Runs, for ONE ReadWindow chunk, the whole HMM filter cascade:
///   F1  SSV/MSV longtarget windows (`f1_msv_windows`, already incl. extend/merge/split)
///   F2  Viterbi   — OFF for this DB size (do_vit=0)
///   F3  local Forward     (cm_pipeline.c:2752-2762)
///   F3b Forward comp bias  (cm_pipeline.c:2772-2784)
/// then merges the surviving windows into non-overlapping ones (2824-2870).
///
/// `chunk_dsq` is the sentinel-padded dsq of one chunk (residues at 1..=win_len);
/// returned window coords are CHUNK-LOCAL 1-based, matching C's ws/we which are
/// relative to sq->dsq. `.score` carries the F3b bit score `wb[i]`.
///
/// Note: F2 Viterbi, F1b/F2b bias, and the `pos_past_*` residue tallies
/// (2794-2822) are intentionally omitted — they don't affect the returned window
/// set for this DB size; they'll be added when the accounting struct is wired.
/// F1(SSV)/F3(Fwd)/F3b(Fwd-bias) per-pass accounting, mirroring the `CM_PLI_ACCT`
/// fields C accumulates in `pli_p7_filter` (cm_pipeline.c:2638, 2765, 2785, 2800-2822).
/// `n_past_msv` = number of MSV windows (`nwin`); `pos_past_*` are the surviving
/// windows' residue extents with adjacent-window overlap subtracted (C 2810-2820).
#[derive(Default, Clone, Copy)]
pub struct F1F3Acct {
    pub n_past_msv: u64,
    pub pos_past_msv: u64,
    /// F1b MSV composition-bias survivors (C survAA[p7_SURV_F1b]).
    pub n_past_msvbias: u64,
    pub pos_past_msvbias: u64,
    /// F2 Viterbi survivors (only tallied when the Viterbi filter is on).
    pub n_past_vit: u64,
    pub pos_past_vit: u64,
    /// F2b Viterbi composition-bias survivors (only tallied when do_vit is on).
    pub n_past_vitbias: u64,
    pub pos_past_vitbias: u64,
    pub n_past_fwd: u64,
    pub pos_past_fwd: u64,
    pub n_past_fwdbias: u64,
    pub pos_past_fwdbias: u64,
}

/// Optional F2 (Viterbi) filter parameters for [`f3_filter_sequence`]. `Some` enables
/// the Viterbi stage between F1 (MSV) and F3 (Forward) — used by the HMM-only pipeline
/// (C `cur_do_vit`, cm_pipeline.c:2706-2716). `None` leaves F2 off (the default <20Mb
/// CM pipeline, where `do_vit`=FALSE), so existing callers are byte-unchanged.
/// Optional F1b (MSV composition-bias) sub-filter parameters for [`f3_filter_sequence`].
/// `Some` enables the MSV-bias stage between F1 (SSV window detection) and F2 (Viterbi)
/// — C `cur_do_msvbias`, cm_pipeline.c:2663-2681, run by `--doF1b`/`--F1b`. `None`
/// leaves it off (the default: `pli->do_msvbias=FALSE`), so existing callers are
/// byte-unchanged. The filter reruns the per-sequence MSV on each window (reconfigured
/// to the window length) and gates by the local MSV Gumbel P-value.
pub struct MsvBiasParams {
    /// F1b P-value threshold (P > f1b ⇒ window dropped).
    pub f1b: f64,
    /// Local MSV Gumbel mu/lambda (p7 CM_p7_LMMU / CM_p7_LMLAMBDA).
    pub lmmu: f64,
    pub lmlambda: f64,
}

pub struct VitParams<'a> {
    pub vf: &'a crate::p7_vitfilter::VitFilter,
    /// F2 P-value threshold (P > f2 ⇒ window dropped).
    pub f2: f64,
    /// C `cur_do_vitbias` (cm_pipeline.c:2726-2740): run the Viterbi composition-bias
    /// sub-filter (F2b) after F2. FALSE for the HMM-only pass; TRUE for the default
    /// CM pipeline (Z_Mb >= 20) and `--rfam`.
    pub do_vitbias: bool,
    /// F2b (Viterbi-bias) P-value threshold (P > f2b ⇒ window dropped).
    pub f2b: f64,
    /// Local Viterbi Gumbel mu/lambda (p7 CM_p7_LVMU / CM_p7_LVLAMBDA).
    pub lvmu: f64,
    pub lvlambda: f64,
}

/// C `esl_gumbel_surv` (esl_gumbel.c:129): P(X>x) for a Gumbel(mu,lambda).
#[inline]
pub fn esl_gumbel_surv(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = lambda * (x - mu);
    let ey = -((-y).exp());
    if ey.abs() < 5e-9 {
        // eslSMALLX1; 1-e^x ~ -x
        -ey
    } else {
        1.0 - ey.exp()
    }
}

pub fn f3_filter_sequence(
    mf: &MsvFilter,
    ff: &ForwardFilter,
    bf: &BiasFilter,
    chunk_dsq: &[u8],
    win_len: usize,
    f1: f64,
    f3: f64,
    f3b: f64,
    cmw: i64,
    msvbias: Option<MsvBiasParams>,
    vit: Option<VitParams>,
    // C cm_pipeline.c:2518-2519 cur_do_fwd/cur_do_fwdbias. For the default CM pipeline
    // both are TRUE. For the HMM-only pass cur_do_fwd = !do_max_hmmonly and
    // cur_do_fwdbias is ALWAYS FALSE (cur_F3b = 1.0). When do_fwd is FALSE (--hmmmax)
    // the Forward filter is skipped entirely and every MSV window survives, exactly as
    // C falls through the `if(cur_do_fwd)` block without a `continue`.
    do_fwd: bool,
    do_fwdbias: bool,
    // C cm_pipeline.c:2514 cur_do_msv. When FALSE (--mid, --max, --nohmm) C does NOT
    // run MSV/SSV window detection; instead it tiles the chunk into deterministic
    // windows of length 2*maxW (cm_pipeline.c:2618-2633). maxw = pli->maxW.
    do_msv: bool,
    maxw: usize,
) -> (Vec<Window>, F1F3Acct) {
    let ln2 = std::f64::consts::LN_2;

    // C 2551-2634: window definition. If cur_do_msv: Filter 1 (SSV longtarget) +
    // ExtendAndMerge + split >2*cmW (all inside f1_msv_windows). Else (--mid etc.):
    // deterministic 2*maxW tiling with (maxW+1)-residue stride (cm_pipeline.c:2618).
    let _tmsv = std::time::Instant::now();
    let wins = if do_msv {
        f1_msv_windows(mf, chunk_dsq, win_len, f1, cmw)
    } else {
        let n = win_len as i64;
        let mw = maxw as i64;
        let mut nwin = 1i64; // first window
        if n > 2 * mw {
            // (L - first window) / (# unique residues per window = maxW+1)
            nwin += (n - 2 * mw) / (mw + 1);
            if (n - 2 * mw) % (mw + 1) > 0 {
                nwin += 1; // add back the fraction the integer division dropped
            }
        }
        (0..nwin)
            .map(|i| {
                let ws = 1 + i * (mw + 1);
                let we = (ws + 2 * mw - 1).min(n);
                Window { start: ws, end: we, score: 0.0 }
            })
            .collect()
    };
    let nwin = wins.len();
    if std::env::var("IX_DEBUG").is_ok() {
        let tot: i64 = wins.iter().map(|w| (w.end - w.start + 1)).sum();
        eprintln!("[F3] f1_msv nwin={} total_win_len={} msv_t={:.2}s (win_len={} cmw={})",
            nwin, tot, _tmsv.elapsed().as_secs_f64(), win_len, cmw);
    }

    // C 2495,2500-2501,2648-2651: per-window bookkeeping.
    //   survAA[p7_SURV_F3b][i] — TRUE if window i survived the whole cascade.
    //   wb[i] = -999.0 initially; overwritten by the furthest-reached filter.
    // C survAA[p7_SURV_F1b]: window reached past the F1b MSV-bias gate. Defaults TRUE
    // (all pass when do_msvbias off); cleared only on an actual F1b drop.
    let mut surv_f1b = vec![true; nwin];
    let mut surv_f2 = vec![true; nwin]; // all survive when F2 is off (do_vit=FALSE)
    // C survAA[p7_SURV_F2b]: window reached past the F2b gate. When do_vit is off this
    // stays all-false (never printed); when on, mirrors surv_f2 minus F2b drops.
    let mut surv_f2b = vec![true; nwin];
    let mut surv_f3 = vec![false; nwin];
    let mut surv_f3b = vec![false; nwin];
    let mut wb = vec![-999.0f32; nwin];

    // C 2653-2792: the per-window loop (cur_do_msvbias/cur_do_vit/cur_do_vitbias
    // all FALSE here, so only F3 and F3b run).
    for (i, w) in wins.iter().enumerate() {
        let ws = w.start;
        let we = w.end;
        let wlen = (we - ws + 1) as usize;

        // C 2654: subdsq = sq->dsq + ws - 1; here we rebuild a sentinel-padded
        // slice holding residues ws..=we at indices 1..=wlen.
        let mut sub = vec![255u8];
        sub.extend_from_slice(&chunk_dsq[ws as usize..=we as usize]);
        sub.push(255u8);

        // C 2658-2659: p7_bg_SetLength + p7_bg_NullOne (length-only geometric null).
        let nullsc = p7_bg_null_one(wlen);

        // C 2663-2681: Filter 1B, MSV composition bias (cur_do_msv && cur_do_msvbias).
        // Rerun the per-sequence MSV on the window (length reconfigured to wlen) to get
        // the full MSV score, subtract the bias-composition null, and gate by the local
        // MSV Gumbel:
        //   p7_oprofile_ReconfigMSVLength(om, wlen); p7_MSVFilter(subdsq, wlen, &mfsc);
        //   p7_bg_FilterScore(bg, subdsq, wlen, &filtersc);
        //   wsc = (mfsc - filtersc)/log2; P = esl_gumbel_surv(wsc, LMMU, LMLAMBDA);
        //   if (P > cur_F1b) continue;  else n_past_msvbias++.
        // An overflowed MSVFilter (eslERANGE) ⇒ very high score ⇒ P≈0 ⇒ survives.
        if let Some(mb) = msvbias.as_ref() {
            if let Some(mfsc) = msv_score_reconfig(mf, &sub, wlen, wlen) {
                let filtersc = bias_filter_score(bf, &sub, wlen);
                let wsc_mb = (mfsc as f64 - filtersc as f64) / ln2;
                let p_mb = esl_gumbel_surv(wsc_mb, mb.lmmu, mb.lmlambda);
                if p_mb > mb.f1b {
                    surv_f1b[i] = false;
                    continue;
                }
            }
        }

        // C 2706-2716: Filter 2, Viterbi (HMM-only pipeline only; cur_do_vit).
        //   p7_ViterbiFilter(subdsq, wlen, om, &vfsc);
        //   wsc = (vfsc - nullsc)/log2; P = esl_gumbel_surv(wsc, LVMU, LVLAMBDA);
        //   if (P > cur_F2) continue;  else n_past_vit++.
        // An overflowed ViterbiFilter (eslERANGE) means a very high score ⇒ survives.
        if let Some(vp) = vit.as_ref() {
            match crate::p7_vitfilter::vit_score(vp.vf, &sub, wlen) {
                Some(vfsc) => {
                    let wsc_v = (vfsc as f64 - nullsc) / ln2;
                    let p_vit = esl_gumbel_surv(wsc_v, vp.lvmu, vp.lvlambda);
                    if p_vit > vp.f2 {
                        surv_f2[i] = false;
                        surv_f2b[i] = false;
                        continue;
                    }
                    // C 2725-2740: Filter 2b, Viterbi composition bias — only when
                    // cur_do_vit && cur_do_vitbias.
                    //   p7_bg_FilterScore(bg, subdsq, wlen, &filtersc);
                    //   wsc = (vfsc - filtersc) / eslCONST_LOG2;
                    //   P   = esl_gumbel_surv(wsc, LVMU, LVLAMBDA);
                    //   if (P > cur_F2b) continue;  else n_past_vitbias++.
                    if vp.do_vitbias {
                        let filtersc = bias_filter_score(bf, &sub, wlen);
                        let wsc_b = (vfsc as f64 - filtersc as f64) / ln2;
                        let p_b = esl_gumbel_surv(wsc_b, vp.lvmu, vp.lvlambda);
                        if p_b > vp.f2b {
                            surv_f2b[i] = false;
                            continue;
                        }
                    }
                }
                // overflow ⇒ vfsc = +inf ⇒ P≈0 for both F2 and F2b ⇒ survives both.
                None => {}
            }
        }

        // C 2747-2757: Filter 3, local Forward — only when cur_do_fwd (line 2747).
        // For --hmmmax (do_fwd==FALSE) C skips this block and the window falls through
        // WITHOUT a `continue`, so it survives F3 unconditionally.
        //   p7_ForwardParser(subdsq, wlen, om, pli->oxf, &fwdsc);
        //   wsc = (fwdsc - nullsc) / eslCONST_LOG2;
        //   P   = esl_exp_surv(wsc, LFTAU, LFLAMBDA);
        //   wb[i] = wsc;
        //   if (P > cur_F3) continue;
        let mut fwdsc = 0.0f32;
        if do_fwd {
            let _tf = std::time::Instant::now();
            fwdsc = forward_filter_score(ff, &sub, wlen);
            if std::env::var("IX_DEBUG").is_ok() {
                let el = _tf.elapsed().as_secs_f64();
                if el > 0.5 { eprintln!("[F3] SLOW forward win#{} wlen={} t={:.2}s", i, wlen, el); }
            }
            let wsc = (fwdsc as f64 - nullsc) / ln2;
            let p = esl_exp_surv(wsc, ff.ftau, ff.flambda);
            wb[i] = wsc as f32;
            if p > f3 {
                continue;
            }
        }
        // C 2759-2760: n_past_fwd++ (unconditional); survAA[p7_SURV_F3][i]=TRUE.
        surv_f3[i] = true;

        // C 2766-2777: Filter 3b, Forward composition bias — only when
        // cur_do_fwd && cur_do_fwdbias (line 2766). In the HMM-only pass cur_do_fwdbias
        // is ALWAYS FALSE (cur_F3b = 1.0), so F3b never runs there.
        //   p7_bg_FilterScore(bg, subdsq, wlen, &filtersc);
        //   wsc = (fwdsc - filtersc) / eslCONST_LOG2;
        //   P   = esl_exp_surv(wsc, LFTAU, LFLAMBDA);
        //   wb[i] = wsc;
        //   if (P > cur_F3b) continue;
        if do_fwd && do_fwdbias {
            let filtersc = bias_filter_score(bf, &sub, wlen);
            let wsc_b = (fwdsc as f64 - filtersc as f64) / ln2;
            let p_b = esl_exp_surv(wsc_b, ff.ftau, ff.flambda);
            wb[i] = wsc_b as f32;
            if p_b > f3b {
                continue;
            }
        }
        // C 2778-2780: n_past_fwdbias++; nsurv_fwd++; survAA[p7_SURV_F3b][i]=TRUE.
        surv_f3b[i] = true;
    }

    // C 2824-2836: create list of just those that survived fwd (F3b).
    let mut new_ws: Vec<i64> = Vec::new();
    let mut new_we: Vec<i64> = Vec::new();
    let mut new_wb: Vec<f32> = Vec::new();
    for i in 0..nwin {
        if surv_f3b[i] {
            new_ws.push(wins[i].start);
            new_we.push(wins[i].end);
            new_wb.push(wb[i]);
        }
    }
    let nsurv_fwd = new_ws.len();

    // C 2838-2851: merge windows that overlap or abut (new_we[i]+1 >= new_ws[i2]),
    //   keeping the higher score.
    //   for(i=0,i2=0; i<nsurv_fwd; i++) {
    //     useme[i]=TRUE; i2=i+1;
    //     while(i2<nsurv_fwd && (new_we[i]+1) >= new_ws[i2]) {
    //       useme[i2]=FALSE; new_we[i]=new_we[i2];
    //       new_wb[i]=ESL_MAX(new_wb[i], new_wb[i2]); i2++;
    //     }
    //     i=i2-1;   /* then for-loop i++ makes i=i2 */
    //   }
    let mut useme = vec![false; nsurv_fwd];
    let mut i = 0usize;
    while i < nsurv_fwd {
        useme[i] = true;
        let mut i2 = i + 1;
        while i2 < nsurv_fwd && (new_we[i] + 1) >= new_ws[i2] {
            useme[i2] = false;
            new_we[i] = new_we[i2];
            new_wb[i] = new_wb[i].max(new_wb[i2]);
            i2 += 1;
        }
        i = i2; // C sets i=i2-1, then the for-loop's i++ yields i=i2.
    }

    // C 2852-2861: compact down to the used (representative) windows.
    let mut out = Vec::new();
    for i in 0..nsurv_fwd {
        if useme[i] {
            out.push(Window {
                start: new_ws[i],
                end: new_we[i],
                score: new_wb[i],
            });
        }
    }

    // C 2794-2822: tally residues surviving each stage, subtracting adjacent-window
    // overlap. survAA[F1][i] is TRUE for ALL windows (they ARE the MSV survivors).
    let mut acct = F1F3Acct::default();
    acct.n_past_msv = nwin as u64; // C 2638: n_past_msv += nwin
    for i in 0..nwin {
        let wlen = (wins[i].end - wins[i].start + 1) as u64;
        acct.pos_past_msv += wlen; // survAA[F1][i] always TRUE
        // C 2681: n_past_msvbias++ / survAA[F1b] — UNCONDITIONAL once a window reaches
        // past the (optional) F1b gate; surv_f1b is TRUE unless an actual F1b drop.
        if surv_f1b[i] {
            acct.n_past_msvbias += 1;
            acct.pos_past_msvbias += wlen;
        }
        // C 2717/2736: n_past_vit++ / n_past_vitbias++ are UNCONDITIONAL once a window
        // reaches that point (survAA[F2]/[F2b]) — they do NOT depend on do_vit/
        // do_vitbias (which only gate whether the filter actually runs, and whether the
        // stat line is displayed). surv_f2/surv_f2b default TRUE and are cleared only on
        // an actual F2/F2b `continue`, so with the Viterbi filter off every window is
        // counted (matching C's --F2b-only display of all MSV windows).
        if surv_f2[i] {
            acct.n_past_vit += 1;
            acct.pos_past_vit += wlen;
        }
        if surv_f2b[i] {
            acct.n_past_vitbias += 1;
            acct.pos_past_vitbias += wlen;
        }
        if surv_f3[i] {
            acct.n_past_fwd += 1;
            acct.pos_past_fwd += wlen;
        }
        if surv_f3b[i] {
            acct.n_past_fwdbias += 1;
            acct.pos_past_fwdbias += wlen;
        }
        if i > 0 {
            let overlap = wins[i - 1].end - wins[i].start + 1; // C 2812
            if overlap > 0 {
                let ov = overlap as u64;
                acct.pos_past_msv -= ov; // F1 always TRUE for both
                if surv_f1b[i] && surv_f1b[i - 1] {
                    acct.pos_past_msvbias -= ov;
                }
                if surv_f2[i] && surv_f2[i - 1] {
                    acct.pos_past_vit -= ov;
                }
                if surv_f2b[i] && surv_f2b[i - 1] {
                    acct.pos_past_vitbias -= ov;
                }
                if surv_f3[i] && surv_f3[i - 1] {
                    acct.pos_past_fwd -= ov;
                }
                if surv_f3b[i] && surv_f3b[i - 1] {
                    acct.pos_past_fwdbias -= ov;
                }
            }
        }
    }
    (out, acct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_match_c_table() {
        let p = CmPipeline::new_default(1.0e6);
        assert_eq!(p.f1, 0.35);
        assert!(!p.do_vit);
        assert_eq!(p.f3, 0.02);
        assert_eq!(p.f6, 0.0001);

        let p = CmPipeline::new_default(5.8e6);
        assert_eq!(p.f1, 0.35);
        assert!(!p.do_vit);
        assert_eq!(p.f3, 0.005);

        let p = CmPipeline::new_default(50.0e6);
        assert!(p.do_vit);
        assert_eq!(p.f3, 0.003);
    }
}
