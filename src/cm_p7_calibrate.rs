//! Faithful port of `cm_p7_Calibrate` (src/cm_p7_modelmaker.c:254) — the p7
//! filter-HMM calibration that cmbuild runs to produce the six E-value
//! statistics written to a `.cm` file:
//!
//! ```text
//! STATS LOCAL MSV      <lmmu>  <lambda>
//! STATS LOCAL VITERBI  <lvmu>  <lambda>
//! STATS LOCAL FORWARD  <lftau> <lambda>
//! EFP7GF               <gfmu>  <gflambda>   (gflambda == lambda)
//! ```
//!
//! Most of `cm_p7_Calibrate` is lifted from HMMER's evalues.c::p7_Calibrate.
//! It uses a SINGLE FAST (LCG) RNG stream, seeded 42, reused across
//! MSVMu → ViterbiMu → p7_Tau → cm_p7_Tau in that exact order. The draw order
//! (one `esl_random` per emitted residue, via `esl_rsq_xfIID`) is the primary
//! byte-parity constraint, so nothing here may reorder or skip a draw.
//!
//! C sources embedded as comments per function. lambda is deterministic
//! (emissions + M only) and can be validated independently of the RNG.

use crate::cm_pipeline::{
    build_forward_filter, build_msv_filter, forward_filter_score, msv_score,
};
use crate::cm_emit::EslRandomFast;
use crate::p7_generic::{build_glocal_profile, p7_gforward, P7Gmx};
use crate::p7_hmm::P7Profile;
use crate::p7_vitfilter::{build_vit_filter, vit_score};
use crate::easel::gumbel::{esl_gumbel_fit_complete, esl_gumbel_fit_complete_loc, esl_gumbel_invcdf};

const LOG2: f64 = std::f64::consts::LN_2; // eslCONST_LOG2

/// C: p7_bg.c::p7_bg_NullOne() after p7_bg_SetLength(bg, L). CRITICAL: `bg->p1`
/// is a **float** = `(float)L/(float)(L+1)`, and the returned `*ret_sc` is a
/// **float** (the `log`s are evaluated in double, then truncated to f32). The
/// filter callers then compute `(sc - nullsc)/log2` with `sc` and `nullsc` both
/// f32, so the subtraction happens in single precision. Getting this precision
/// wrong shifts every Forward-based tau in the 4th decimal (the byte-quantized
/// MSV/Viterbi scores are insensitive to it, but Forward scores are not).
/// ```c
/// bg->p1 = (float) L / (float) (L+1);
/// *ret_sc = (float) L * log(bg->p1) + log(1. - bg->p1);
/// ```
#[inline]
fn p7_bg_null_one_f32(l: usize) -> f32 {
    let p1 = (l as f32) / ((l + 1) as f32);
    ((l as f64) * (p1 as f64).ln() + (1.0 - p1 as f64).ln()) as f32
}

/// The six calibration numbers. `lambda` is shared by the three local stats and
/// by EFP7GF (`gflambda == lambda`). Also written into `hmm.evparam`.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationResult {
    pub lmmu: f64,
    pub lvmu: f64,
    pub lftau: f64,
    pub lambda: f64,
    pub gfmu: f64,
    pub gflambda: f64,
}

/// C: esl_randomseq.c::esl_rnd_FChoose() — sample a residue 0..K-1 from float
/// distribution `p`, in double precision. One `esl_random` draw per call.
/// ```c
/// double norm = 0.0, sum = 0.0, roll = esl_random(r);
/// for (i=0;i<N;i++) norm += p[i];
/// for (i=0;i<N;i++) { sum += (double) p[i]; if (roll < (sum / norm)) return i; }
/// ```
#[inline]
fn esl_rnd_fchoose(r: &mut EslRandomFast, p: &[f32]) -> usize {
    let roll = r.random();
    let mut norm = 0.0f64;
    for &pi in p {
        norm += pi as f64;
    }
    let mut sum = 0.0f64;
    for (i, &pi) in p.iter().enumerate() {
        sum += pi as f64;
        if roll < (sum / norm) {
            return i;
        }
    }
    p.len() - 1 // C esl_fatal's here (unreached); keep total residue count intact
}

/// C: esl_randomseq.c::esl_rsq_xfIID() — fill a digital sequence of length L with
/// i.i.d. residues from `p`. `dsq` is (L+2) long; `[0]` and `[L+1]` are
/// sentinels. Exactly L `esl_random` draws, in order.
/// ```c
/// dsq[0] = dsq[L+1] = eslDSQ_SENTINEL;
/// for (x = 1; x <= L; x++) dsq[x] = esl_rnd_FChoose(r, p, K);
/// ```
fn esl_rsq_xfiid(r: &mut EslRandomFast, p: &[f32], l: usize, dsq: &mut [u8]) {
    dsq[0] = 255; // eslDSQ_SENTINEL
    dsq[l + 1] = 255;
    for x in 1..=l {
        dsq[x] = esl_rnd_fchoose(r, p) as u8;
    }
}

/// C: esl_vectorops.c::esl_vec_FRelEntropy() over one match state, then averaged
/// in p7_MeanMatchRelativeEntropy (modelstats.c:80). Accumulator `kl` is a
/// float; the per-state term uses double `log2`.
/// ```c
/// kl = 0.;
/// for (i=0;i<n;i++) if (p[i] > 0.) { if (q[i]==0.) return inf; else kl += p[i]*log2(p[i]/q[i]); }
/// ... KL += esl_vec_FRelEntropy(hmm->mat[k], bg->f, K);  KL /= M;
/// ```
fn p7_mean_match_relative_entropy(hmm: &P7Profile) -> f64 {
    let m = hmm.m as usize;
    let mut kl_total = 0.0f64;
    for k in 1..=m {
        let mut kl: f32 = 0.0;
        for i in 0..4 {
            let pv = hmm.mat[k][i];
            if pv > 0.0 {
                // q[i] = bg->f[i] = 0.25 (uniform RNA background), never 0.
                let ratio = pv / 0.25f32; // p[i]/q[i] in float
                kl = ((kl as f64) + (pv as f64) * (ratio as f64).log2()) as f32;
            }
        }
        kl_total += kl as f64;
    }
    kl_total / m as f64
}

/// C: evalues.c::p7_Lambda() (184).
/// ```c
/// double H = p7_MeanMatchRelativeEntropy(hmm, bg);
/// *ret_lambda = eslCONST_LOG2 + 1.44 / ((double) hmm->M * H);
/// ```
fn p7_lambda(hmm: &P7Profile) -> f64 {
    let h = p7_mean_match_relative_entropy(hmm);
    LOG2 + 1.44 / (hmm.m as f64 * h)
}

/// C: evalues.c::p7_MSVMu() (238). Simulate N seqs of length L, score with the
/// MSV filter, fit a Gumbel of known lambda for its location mu.
/// ```c
/// float maxsc = (255 - om->base_b) / om->scale_b;
/// p7_oprofile_ReconfigLength(om, L); p7_bg_SetLength(bg, L);
/// for (i=0;i<N;i++) {
///   esl_rsq_xfIID(r, bg->f, K, L, dsq);
///   p7_bg_NullOne(bg, dsq, L, &nullsc);
///   status = p7_MSVFilter(dsq, L, om, ox, &sc);
///   if (status == eslERANGE) sc = maxsc;
///   xv[i] = (sc - nullsc) / eslCONST_LOG2;
/// }
/// esl_gumbel_FitCompleteLoc(xv, N, lambda, ret_mmu);
/// ```
fn p7_msv_mu(r: &mut EslRandomFast, hmm: &P7Profile, bgf: &[f32], l: usize, n: usize, lambda: f64) -> f64 {
    // build_msv_filter(p7, L) sets tjb_b from L, i.e. equivalent to
    // p7_oprofile_ReconfigLength(om, L) for the MSV special states.
    let mf = build_msv_filter(hmm, l);
    let maxsc = mf.cal_maxsc();
    let nullsc = p7_bg_null_one_f32(l); // p7_bg_SetLength(bg,L) then NullOne (f32)
    let mut dsq = vec![0u8; l + 2];
    let mut xv = vec![0.0f64; n];
    for xi in xv.iter_mut() {
        esl_rsq_xfiid(r, bgf, l, &mut dsq);
        let sc = match msv_score(&mf, &dsq, l) {
            Some(s) => s,
            None => maxsc, // eslERANGE
        };
        *xi = ((sc - nullsc) as f64) / LOG2; // C: (sc - nullsc)/eslCONST_LOG2, f32 sub
    }
    esl_gumbel_fit_complete_loc(&xv, lambda).unwrap_or(0.0)
}

/// C: evalues.c::p7_ViterbiMu() (307). Identical to p7_MSVMu but with the
/// Viterbi filter and `maxsc = (32767 - base_w)/scale_w`.
fn p7_viterbi_mu(r: &mut EslRandomFast, hmm: &P7Profile, bgf: &[f32], l: usize, n: usize, lambda: f64) -> f64 {
    let vf = build_vit_filter(hmm);
    let maxsc = vf.cal_maxsc();
    let nullsc = p7_bg_null_one_f32(l);
    let mut dsq = vec![0u8; l + 2];
    let mut xv = vec![0.0f64; n];
    for xi in xv.iter_mut() {
        esl_rsq_xfiid(r, bgf, l, &mut dsq);
        let sc = match vit_score(&vf, &dsq, l) {
            Some(s) if s.is_finite() => s,
            Some(_) => maxsc, // -inf shouldn't occur; treat as range guard
            None => maxsc,    // eslERANGE
        };
        *xi = ((sc - nullsc) as f64) / LOG2; // C: (sc - nullsc)/eslCONST_LOG2, f32 sub
    }
    esl_gumbel_fit_complete_loc(&xv, lambda).unwrap_or(0.0)
}

/// C: evalues.c::p7_Tau() (412). Local Forward. Fit a Gumbel to N Forward scores,
/// then extrapolate the tail at mass `tailp`.
/// ```c
/// for (i=0;i<N;i++) {
///   esl_rsq_xfIID(r, bg->f, K, L, dsq);
///   p7_ForwardParser(dsq, L, om, ox, &fsc);
///   p7_bg_NullOne(bg, dsq, L, &nullsc);
///   xv[i] = (fsc - nullsc) / eslCONST_LOG2;
/// }
/// esl_gumbel_FitComplete(xv, N, &gmu, &glam);
/// *ret_tau = esl_gumbel_invcdf(1.0-tailp, gmu, glam) + (log(tailp) / lambda);
/// ```
fn p7_tau(r: &mut EslRandomFast, hmm: &P7Profile, bgf: &[f32], l: usize, n: usize, lambda: f64, tailp: f64) -> f64 {
    let ff = build_forward_filter(hmm); // p7_ProfileConfig LOCAL multihit + fb_conversion
    let nullsc = p7_bg_null_one_f32(l);
    let mut dsq = vec![0u8; l + 2];
    let mut xv = vec![0.0f64; n];
    for xi in xv.iter_mut() {
        esl_rsq_xfiid(r, bgf, l, &mut dsq);
        let fsc = forward_filter_score(&ff, &dsq, l); // nats; ReconfigLength(l) done inside
        *xi = ((fsc - nullsc) as f64) / LOG2; // C: (fsc - nullsc)/eslCONST_LOG2, f32 sub
    }
    let (gmu, glam) = esl_gumbel_fit_complete(&xv).unwrap_or((0.0, 0.0));
    esl_gumbel_invcdf(1.0 - tailp, gmu, glam) + (tailp.ln() / lambda)
}

/// C: cm_p7_modelmaker.c::cm_p7_Tau() (344). Glocal generic Forward variant of
/// p7_Tau. Same fit/extrapolation, but scores come from `p7_GForward` on a
/// GLOCAL generic profile.
/// ```c
/// p7_ReconfigLength(gm, L); p7_bg_SetLength(bg, L);
/// for (i=0;i<N;i++) {
///   esl_rsq_xfIID(r, bg->f, K, L, dsq);
///   p7_GForward(dsq, L, gm, gx, &fsc);
///   p7_bg_NullOne(bg, dsq, L, &nullsc);
///   sc = (fsc - nullsc) / eslCONST_LOG2; xv[i] = sc;
/// }
/// esl_gumbel_FitComplete(xv, N, &gmu, &glam);
/// *ret_tau = esl_gumbel_invcdf(1.0-tailp, gmu, glam) + (log(tailp) / lambda);
/// ```
fn cm_p7_tau(r: &mut EslRandomFast, hmm: &P7Profile, bgf: &[f32], l: usize, n: usize, lambda: f64, tailp: f64) -> f64 {
    // p7_ProfileConfig(hmm, bg, gm, L, p7_GLOCAL) + p7_ReconfigLength(gm, L).
    let gm = build_glocal_profile(hmm, l as i32);
    let mut gx = P7Gmx::new(gm.m, l);
    let nullsc = p7_bg_null_one_f32(l);
    let mut dsq = vec![0u8; l + 2];
    let mut xv = vec![0.0f64; n];
    for xi in xv.iter_mut() {
        esl_rsq_xfiid(r, bgf, l, &mut dsq);
        let fsc = p7_gforward(&dsq, l, &gm, &mut gx); // nats
        *xi = ((fsc - nullsc) as f64) / LOG2; // C: (fsc - nullsc)/eslCONST_LOG2, f32 sub
    }
    let (gmu, glam) = esl_gumbel_fit_complete(&xv).unwrap_or((0.0, 0.0));
    esl_gumbel_invcdf(1.0 - tailp, gmu, glam) + (tailp.ln() / lambda)
}

/// Faithful port of `cm_p7_Calibrate` (cm_p7_modelmaker.c:254), with the
/// cmbuild default arguments (build_and_calibrate_p7_filter, cmbuild.c:2293):
///
/// * ElmL = ElvL = 200, ElfL = 100, EgfL = max(100, 2*clen)
/// * ElmN = ElvN = ElfN = EgfN = 200  (--EmN/--EvN/--ElfN/--EgfN defaults)
/// * lftailp = 0.055 (--Elftp), gftailp = 0.065 (--Egftp)   [cmbuild.c:116-117]
///
/// A single FAST/LCG RNG seeded 42 is threaded through MSVMu → ViterbiMu →
/// p7_Tau → cm_p7_Tau in that order. On return `hmm.evparam` is populated
/// (MMU/MLAMBDA/VMU/VLAMBDA/FTAU/FLAMBDA) exactly like the C, and the six numbers
/// are returned (EFP7GF = (gfmu, lambda)).
/// Number of sampled seqs for each p7-filter-HMM calibration fit
/// (--EmN/--EvN/--ElfN/--EgfN, cmbuild.c:111-114; all default 200).
#[derive(Clone, Copy, Debug)]
pub struct P7CalN {
    pub emn: usize,  // --EmN: local MSV Gumbel mu fit
    pub evn: usize,  // --EvN: local Vit Gumbel mu fit
    pub elfn: usize, // --ElfN: local Fwd Gumbel mu fit
    pub egfn: usize, // --EgfN: glocal Fwd Gumbel mu fit
}
impl Default for P7CalN {
    fn default() -> Self {
        P7CalN { emn: 200, evn: 200, elfn: 200, egfn: 200 }
    }
}

pub fn cm_p7_calibrate(hmm: &mut P7Profile, clen: i32, ns: P7CalN) -> CalibrationResult {
    // cmbuild.c:2296-2303 default parameters.
    let elm_l = 200usize; // lmsvL
    let elv_l = 200usize; // lvitL
    let elf_l = 100usize; // lfwdL
    let egf_l = std::cmp::max(100i32, 2 * clen) as usize; // gfwdL
    // Sample counts, from --EmN/--EvN/--ElfN/--EgfN (cmbuild.c:2299-2302).
    let elm_n = ns.emn;
    let elv_n = ns.evn;
    let elf_n = ns.elfn;
    let egf_n = ns.egfn;
    let lftailp = 0.055f64; // --Elftp default
    let gftailp = 0.065f64; // --Egftp default

    // r = esl_randomness_CreateFast(42); bg = p7_bg_Create(RNA) => f[x] = 0.25.
    let mut r = EslRandomFast::new(42);
    let bgf = [0.25f32; 4];

    // p7_Lambda (deterministic; no RNG).
    let lambda = p7_lambda(hmm);

    // The RNG-driven simulations, in the C order (single stream).
    let lmmu = p7_msv_mu(&mut r, hmm, &bgf, elm_l, elm_n, lambda);
    let lvmu = p7_viterbi_mu(&mut r, hmm, &bgf, elv_l, elv_n, lambda);
    let lftau = p7_tau(&mut r, hmm, &bgf, elf_l, elf_n, lambda, lftailp);

    // Set the p7's evparam[] (cm_p7_modelmaker.c:286-292).
    hmm.evparam.lmmu = lmmu;
    hmm.evparam.lmlambda = lambda;
    hmm.evparam.lvmu = lvmu;
    hmm.evparam.lvlambda = lambda;
    hmm.evparam.lftau = lftau;
    hmm.evparam.lflambda = lambda;

    // Glocal Forward stats (reuses the same RNG stream).
    let gfmu = cm_p7_tau(&mut r, hmm, &bgf, egf_l, egf_n, lambda, gftailp);
    let gflambda = lambda;

    CalibrationResult {
        lmmu,
        lvmu,
        lftau,
        lambda,
        gfmu,
        gflambda,
    }
}
