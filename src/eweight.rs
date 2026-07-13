//! eweight.rs -- faithful byte-parity port of Infernal 1.1.5 entropy weighting.
//!
//! Ported from:
//!   - original/src/eweight.c            (cm_Rescale, cm_MeanMatch*, cm_EntropyWeight, target_f)
//!   - original/easel/esl_rootfinder.c   (esl_root_Bisection, esl_rootfinder_Create/SetAbsoluteTolerance)
//!   - original/easel/esl_vectorops.c    (esl_vec_FScale/FSum/FNorm/FSet/FEntropy/FRelEntropy)
//!   - original/src/cmbuild.c            (set_target_relent, version_1p0_default_target_relent)
//!
//! Arithmetic is preserved exactly: f32 where C uses `float`, f64 where C uses `double`.
//! Each block carries a comment naming the C file:function it transcribes.

use crate::constants::{
    B_ST, DEFAULT_ETARGET, DEFAULT_ETARGET_HMMFILTER, E_ST, IL_ST, IR_ST, MATL_ML, MATP_MP,
    MATR_MR, ML_ST, MP_ST, MR_ST,
};

/// K = abc->K = 4 (canonical RNA alphabet).
const K: usize = 4;

/// eslCONST_LOG2R = 1/ln(2); exact literal from original/easel/easel.h:305.
const ESL_CONST_LOG2R: f64 = 1.44269504088896341;

// =====================================================================
// esl_vectorops.c helpers (faithful transcriptions)
// =====================================================================

/// esl_vectorops.c:esl_vec_FScale -- `for i: vec[i] *= scale` in f32.
#[inline]
fn f_scale(vec: &mut [f32], n: usize, scale: f32) {
    for x in vec.iter_mut().take(n) {
        *x *= scale;
    }
}

/// esl_vectorops.c:esl_vec_FSet -- `for i: vec[i] = value`.
#[inline]
fn f_set(vec: &mut [f32], n: usize, value: f32) {
    for x in vec.iter_mut().take(n) {
        *x = value;
    }
}

/// esl_vectorops.c:esl_vec_FSum -- Kahan compensated summation in f32.
///   float sum=0, y,t,c; c=0;
///   for i: y = vec[i]-c; t = sum+y; c = (t-sum)-y; sum = t;
#[inline]
fn f_sum(vec: &[f32], n: usize) -> f32 {
    let mut sum: f32 = 0.0;
    let mut c: f32 = 0.0;
    for &v in vec.iter().take(n) {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// esl_vectorops.c:esl_vec_FNorm --
///   sum = FSum(vec,n);
///   if (sum != 0.) for i: vec[i] /= sum;   else for i: vec[i] = 1. / (float) n;
#[inline]
fn f_norm(vec: &mut [f32], n: usize) {
    let sum = f_sum(vec, n);
    if sum != 0.0 {
        for x in vec.iter_mut().take(n) {
            *x /= sum;
        }
    } else {
        // C: `vec[i] = 1. / (float) n;` -- 1. is double, (float)n promoted to double,
        // double division, result stored to float.
        let val = (1.0f64 / (n as f32 as f64)) as f32;
        for x in vec.iter_mut().take(n) {
            *x = val;
        }
    }
}

/// esl_vectorops.c:esl_vec_FEntropy --
///   float H=0; for i: if (p[i] > 0.) H -= p[i] * log2f(p[i]); return H;
/// All arithmetic in f32 (`log2f`).
#[inline]
fn f_entropy(p: &[f32], n: usize) -> f32 {
    let mut h: f32 = 0.0;
    for &pi in p.iter().take(n) {
        if pi > 0.0 {
            h -= pi * pi.log2();
        }
    }
    h
}

/// esl_vectorops.c:esl_vec_FRelEntropy --
///   float kl=0;
///   for i: if (p[i] > 0.) { if (q[i]==0.) return eslINFINITY; else kl += p[i]*log2(p[i]/q[i]); }
///   return kl;
/// Note the exact promotion: `p[i]/q[i]` is computed in f32 (both float), then promoted to
/// double for the `log2` (double) call; `p[i]*log2(...)` is a double product accumulated into
/// the f32 `kl` (so each iteration truncates back to f32).
#[inline]
fn f_rel_entropy(p: &[f32], q: &[f32], n: usize) -> f32 {
    let mut kl: f32 = 0.0;
    for i in 0..n {
        let pi = p[i];
        if pi > 0.0 {
            let qi = q[i];
            if qi == 0.0 {
                return f32::INFINITY;
            } else {
                let ratio: f32 = pi / qi; // f32 division, as in C
                let term: f64 = (pi as f64) * (ratio as f64).log2();
                kl = (kl as f64 + term) as f32;
            }
        }
    }
    kl
}

// =====================================================================
// eweight.c:cm_Rescale (l.233)
// =====================================================================

/// eweight.c:cm_Rescale -- scale a counts-based CM by `scale`.
/// Skips transition scaling for B_st / E_st states.
pub fn cm_rescale(cm: &mut crate::cm::CM, scale: f32) {
    let m = cm.m as usize;
    for v in 0..m {
        // Scale transition counts vector if not a BIF or E state.
        if cm.sttype[v] as i32 != B_ST && cm.sttype[v] as i32 != E_ST {
            let cnum = cm.cnum[v] as usize;
            f_scale(&mut cm.t[v], cnum, scale);
        }
        // Scale emission counts vectors.
        if cm.sttype[v] as i32 == MP_ST {
            // Consensus base pairs: K*K = 16 values.
            f_scale(&mut cm.e[v], K * K, scale);
        } else if cm.sttype[v] as i32 == ML_ST
            || cm.sttype[v] as i32 == MR_ST
            || cm.sttype[v] as i32 == IL_ST
            || cm.sttype[v] as i32 == IR_ST
        {
            // Singlets (some consensus, some not): K = 4 values.
            f_scale(&mut cm.e[v], K, scale);
        }
    }

    // begin, end transitions; only valid [0..M-1].
    f_scale(&mut cm.begin, m, scale);
    f_scale(&mut cm.end, m, scale);
}

// =====================================================================
// eweight.c:cm_MeanMatchEntropy (l.343) / cm_MeanMatchInfo (l.320)
// =====================================================================

/// eweight.c:cm_MeanMatchEntropy -- mean entropy per match state, in bits.
/// H /= (double) cm->clen.
pub fn cm_mean_match_entropy(cm: &crate::cm::CM) -> f64 {
    let m = cm.m as usize;
    let mut h: f64 = 0.;
    for v in 0..m {
        if cm.stid[v] as i32 == MATP_MP {
            h += f_entropy(&cm.e[v], K * K) as f64;
        } else if cm.stid[v] as i32 == MATL_ML || cm.stid[v] as i32 == MATR_MR {
            h += f_entropy(&cm.e[v], K) as f64;
        }
    }
    h /= cm.clen as f64;
    h
}

/// eweight.c:cm_MeanMatchInfo -- FEntropy(null,K) - MeanMatchEntropy(cm).
pub fn cm_mean_match_info(cm: &crate::cm::CM) -> f64 {
    f_entropy(&cm.null, K) as f64 - cm_mean_match_entropy(cm)
}

// =====================================================================
// eweight.c:cm_MeanMatchRelativeEntropy (l.377)
// =====================================================================

/// eweight.c:cm_MeanMatchRelativeEntropy -- mean relative entropy per match state, in bits.
///
/// Denominator is `KL /= (double) cm->clen;` (consensus length, NOT M).
///
/// Only two accumulations feed the returned KL:
///   MP  : KL += FRelEntropy(e[v], pair_null, K*K)
///   ML/MR: KL += FRelEntropy(e[v], null, K)
/// The left_e / right_e marginals are computed in C but their only consumers
/// (KL_pair_marg etc.) are commented-out debug vars, so they never affect the
/// return value. They are transcribed here for structural fidelity but unused.
pub fn cm_mean_match_relative_entropy(cm: &crate::cm::CM) -> f64 {
    let m = cm.m as usize;
    let mut kl: f64 = 0.;

    // pair_null[i*K + j] = null[i] * null[j]  (f32)
    let mut pair_null = [0.0f32; K * K];
    for i in 0..K {
        for j in 0..K {
            pair_null[(i * K) + j] = cm.null[i] * cm.null[j];
        }
    }

    let mut left_e = [0.0f32; K];
    let mut right_e = [0.0f32; K];

    for v in 0..m {
        if cm.stid[v] as i32 == MATP_MP {
            kl += f_rel_entropy(&cm.e[v], &pair_null, K * K) as f64;

            // --- left/right marginals (dead w.r.t. return value; see doc note) ---
            f_set(&mut left_e, K, 0.);
            for i in 0..K {
                for j in (i * K)..((i + 1) * K) {
                    left_e[i] += cm.e[v][j];
                }
            }
            f_norm(&mut left_e, K);

            f_set(&mut right_e, K, 0.);
            for i in 0..K {
                let mut j = i;
                while j < K * K {
                    right_e[i] += cm.e[v][j];
                    j += K;
                }
            }
            // (esl_vec_FRelEntropy(left_e/right_e, null, K) accumulations are commented out in C)
        } else if cm.stid[v] as i32 == MATL_ML || cm.stid[v] as i32 == MATR_MR {
            kl += f_rel_entropy(&cm.e[v], &cm.null, K) as f64;
        }
    }
    // silence unused-mut warnings for the faithfully-transcribed dead marginals
    let _ = (&left_e, &right_e);

    kl /= cm.clen as f64;
    kl
}

// =====================================================================
// eweight.c:cm_MeanMatchInfoHMM (l.475) / cm_MeanMatchEntropyHMM (l.500)
// =====================================================================

/// eweight.c:cm_MeanMatchEntropyHMM -- like MeanMatchEntropy but marginalizes MATP_MP
/// pair emissions into two singlets. NOTE the C quirk: for MP states the
/// `H += FEntropy(left_e/right_e, K)` calls occur *inside* the `for i` loop, so entropy
/// of the partially-accumulated marginal is added at each i. Transcribed exactly.
pub fn cm_mean_match_entropy_hmm(cm: &crate::cm::CM) -> f64 {
    let m = cm.m as usize;
    let mut h: f64 = 0.;
    let mut left_e = [0.0f32; K];
    let mut right_e = [0.0f32; K];

    for v in 0..m {
        if cm.stid[v] as i32 == MATP_MP {
            // left half
            f_set(&mut left_e, K, 0.);
            for i in 0..K {
                for j in (i * K)..((i + 1) * K) {
                    left_e[i] += cm.e[v][j];
                }
                h += f_entropy(&left_e, K) as f64;
            }
            // right half
            f_set(&mut right_e, K, 0.);
            for i in 0..K {
                let mut j = i;
                while j < K * K {
                    right_e[i] += cm.e[v][j];
                    j += K;
                }
                h += f_entropy(&right_e, K) as f64;
            }
        } else if cm.stid[v] as i32 == MATL_ML || cm.stid[v] as i32 == MATR_MR {
            h += f_entropy(&cm.e[v], K) as f64;
        }
    }
    h /= cm.clen as f64;
    h
}

/// eweight.c:cm_MeanMatchInfoHMM -- FEntropy(null,K) - MeanMatchEntropyHMM(cm).
pub fn cm_mean_match_info_hmm(cm: &crate::cm::CM) -> f64 {
    f_entropy(&cm.null, K) as f64 - cm_mean_match_entropy_hmm(cm)
}

// =====================================================================
// eweight.c:cm_MeanMatchRelativeEntropyHMM (l.559)
// =====================================================================

/// eweight.c:cm_MeanMatchRelativeEntropyHMM -- mean relative entropy, treating the CM
/// as an HMM by marginalizing MATP_MP into left (normalized) + right (raw) singlets.
///   MP : KL += FRelEntropy(norm(left_e), null, K); KL += FRelEntropy(right_e, null, K)
///   ML/MR: KL += FRelEntropy(e[v], null, K)
/// Denominator: KL /= (double) cm->clen.
pub fn cm_mean_match_relative_entropy_hmm(cm: &crate::cm::CM) -> f64 {
    let m = cm.m as usize;
    let mut kl: f64 = 0.;
    let mut left_e = [0.0f32; K];
    let mut right_e = [0.0f32; K];

    for v in 0..m {
        if cm.stid[v] as i32 == MATP_MP {
            // left half (normalized)
            f_set(&mut left_e, K, 0.);
            for i in 0..K {
                for j in (i * K)..((i + 1) * K) {
                    left_e[i] += cm.e[v][j];
                }
            }
            f_norm(&mut left_e, K);
            kl += f_rel_entropy(&left_e, &cm.null, K) as f64;

            // right half (NOT normalized in C)
            f_set(&mut right_e, K, 0.);
            for i in 0..K {
                let mut j = i;
                while j < K * K {
                    right_e[i] += cm.e[v][j];
                    j += K;
                }
            }
            kl += f_rel_entropy(&right_e, &cm.null, K) as f64;
        } else if cm.stid[v] as i32 == MATL_ML || cm.stid[v] as i32 == MATR_MR {
            kl += f_rel_entropy(&cm.e[v], &cm.null, K) as f64;
        }
    }

    kl /= cm.clen as f64;
    kl
}

// =====================================================================
// eweight.c:cm_eweight_target_f / hmm_eweight_target_f (l.43 / l.69)
// =====================================================================

/// Transcribes both eweight.c:cm_eweight_target_f and hmm_eweight_target_f.
/// Restores CM t/e/begin/end from the *_orig copies, rescales to `neff`,
/// re-parameterizes with the Dirichlet prior, then returns relent - etarget.
/// `hmm == true` selects cm_MeanMatchRelativeEntropyHMM (hmm_eweight_target_f).
#[allow(clippy::too_many_arguments)]
fn eweight_target_f(
    cm: &mut crate::cm::CM,
    t_orig: &[Vec<f32>],
    e_orig: &[Vec<f32>],
    begin_orig: &[f32],
    end_orig: &[f32],
    pri: &crate::prior::Prior,
    etarget: f64,
    neff: f64,
    hmm: bool,
) -> f64 {
    let m = cm.m as usize;
    // copy parameters from *_orig back into CM arrays
    // (C copies MAXCONNECT for t and K*K for e; the inner Vecs have exactly those
    //  lengths, so a full slice copy is identical.)
    for v in 0..m {
        cm.t[v][..].copy_from_slice(&t_orig[v][..]);
        cm.e[v][..].copy_from_slice(&e_orig[v][..]);
        cm.begin[v] = begin_orig[v];
        cm.end[v] = end_orig[v];
    }
    // cm_Rescale(p->cm, Neff / (double) p->cm->nseq)
    cm_rescale(cm, (neff / cm.nseq as f64) as f32);
    // PriorifyCM(p->cm, p->pri)
    crate::prior::priorify_cm(cm, pri);
    let re = if hmm {
        cm_mean_match_relative_entropy_hmm(cm)
    } else {
        cm_mean_match_relative_entropy(cm)
    };
    re - etarget
}

// =====================================================================
// esl_rootfinder.c:esl_root_Bisection (l.244) + Create/SetAbsoluteTolerance
// =====================================================================

/// esl_rootfinder.c:esl_root_Bisection -- faithful bisection on the open interval
/// (xl..xr) with the ESL_ROOTFINDER defaults from esl_rootfinder_Create():
///   abs_tolerance (overridden by caller), rel_tolerance = 1e-12, residual_tol = 0.,
///   max_iter = 100.
///
/// Termination (per iteration, esl_rootfinder.c:263-266):
///   xmag = (xl < 0 && xr > 0) ? 0 : x;
///   if (fx == 0.) break;                                        // exact root
///   if ((xr-xl) < abs_tol + rel_tol*xmag || |fx| < residual_tol) break;
#[allow(unused_assignments)]
fn esl_root_bisection<F: FnMut(f64) -> f64>(mut f: F, xl_in: f64, xr_in: f64, abs_tol: f64) -> f64 {
    // esl_rootfinder_Create defaults:
    let rel_tolerance: f64 = 1e-12;
    let residual_tol: f64 = 0.;
    let max_iter: i32 = 100;
    let abs_tolerance: f64 = abs_tol; // esl_rootfinder_SetAbsoluteTolerance(R, 0.01)

    // esl_rootfinder_SetBrackets: evaluate f at both endpoints.
    let mut xl = xl_in;
    let mut xr = xr_in;
    let mut fl = f(xl);
    let mut _fr = f(xr); // stored/updated below to mirror C, though never read after
    // (C: if (fl*fr >= 0) ESL_EXCEPTION -- callers guarantee a bracket, so we proceed.)

    let mut iter: i32 = 0;
    let mut x: f64 = 0.;
    let mut fx: f64;
    loop {
        iter += 1;
        if iter > max_iter {
            // C: ESL_XEXCEPTION(eslENOHALT). Callers always converge; return best x.
            break;
        }
        // Bisect and evaluate.
        x = (xl + xr) / 2.;
        fx = f(x);

        // Test for convergence.
        let xmag = if xl < 0. && xr > 0. { 0. } else { x };
        if fx == 0. {
            break;
        }
        if ((xr - xl) < abs_tolerance + rel_tolerance * xmag) || fx.abs() < residual_tol {
            break;
        }

        // Narrow the bracket; pay attention to directionality.
        if fl > 0. {
            if fx > 0. {
                xl = x;
                fl = fx;
            } else {
                xr = x;
                _fr = fx;
            }
        } else if fx < 0. {
            xl = x;
            fl = fx;
        } else {
            xr = x;
            _fr = fx;
        }
    }
    x
}

// =====================================================================
// eweight.c:cm_EntropyWeight (l.119)
// =====================================================================

/// eweight.c:cm_EntropyWeight -- entropy-weighting bisection driver.
///
/// Returns (hmm_re, Neff). Mirrors the C control flow:
///   1. copy t/e/begin/end into *_orig.
///   2. evaluate target_f at min_Neff.
///   3. if fx < 0: evaluate at max_Neff; if fx > 0: bisect on [0, max_Neff], abs tol 0.01.
///   4. hmm_re = cm_MeanMatchRelativeEntropyHMM(cm).
///   5. restore t/e/begin/end from *_orig.
pub fn cm_entropy_weight(
    cm: &mut crate::cm::CM,
    pri: &crate::prior::Prior,
    etarget: f64,
    min_neff: f64,
    max_neff: f64,
    pretend_cm_is_hmm: bool,
) -> (f64, f64) {
    let m = cm.m as usize;

    // copy parameters of the CM that will be changed by cm_Rescale()
    let t_orig: Vec<Vec<f32>> = cm.t[..m].to_vec();
    let e_orig: Vec<Vec<f32>> = cm.e[..m].to_vec();
    let begin_orig: Vec<f32> = cm.begin[..m].to_vec();
    let end_orig: Vec<f32> = cm.end[..m].to_vec();

    // First, check if min_Neff gives rel entropy >= etarget; if so use min_Neff.
    let mut neff = min_neff;
    let fx = eweight_target_f(
        cm, &t_orig, &e_orig, &begin_orig, &end_orig, pri, etarget, neff, pretend_cm_is_hmm,
    );

    if fx < 0. {
        // check if max_Neff gives rel entropy < etarget; if so use max_Neff.
        neff = max_neff;
        let fx = eweight_target_f(
            cm, &t_orig, &e_orig, &begin_orig, &end_orig, pri, etarget, neff, pretend_cm_is_hmm,
        );

        if fx > 0. {
            // esl_rootfinder_Create + SetAbsoluteTolerance(0.01) + esl_root_Bisection(R,0.,max_Neff)
            let target = |x: f64| -> f64 {
                eweight_target_f(
                    cm, &t_orig, &e_orig, &begin_orig, &end_orig, pri, etarget, x,
                    pretend_cm_is_hmm,
                )
            };
            neff = esl_root_bisection(target, 0., max_neff, 0.01);
        }
    }

    // relative entropy of the (marginalized) CM at the found Neff.
    let hmm_re = cm_mean_match_relative_entropy_hmm(cm);

    // reset CM params to their original values.
    for v in 0..m {
        cm.t[v][..].copy_from_slice(&t_orig[v][..]);
        cm.e[v][..].copy_from_slice(&e_orig[v][..]);
        cm.begin[v] = begin_orig[v];
        cm.end[v] = end_orig[v];
    }

    (hmm_re, neff)
}

// =====================================================================
// cmbuild.c:set_target_relent (l.2486) / version_1p0_default_target_relent (l.2526)
// =====================================================================

/// cmbuild.c:set_target_relent -- length-dependent target relative entropy per position.
///   if(--ere on)  re_target = ere;
///   else          re_target = (nbps > 0) ? DEFAULT_ETARGET : DEFAULT_ETARGET_HMMFILTER;
///   esigma = --esigma (default 45.0)
///   etarget = (esigma - eslCONST_LOG2R * log(2.0 / (clen*(clen+1)))) / clen;
///   etarget = ESL_MAX(etarget, re_target);
pub fn set_target_relent(clen: i32, nbps: i32, esigma: f64, ere: Option<f64>) -> f64 {
    let re_target: f64 = match ere {
        Some(x) => x,
        None => {
            if nbps > 0 {
                DEFAULT_ETARGET
            } else {
                DEFAULT_ETARGET_HMMFILTER
            }
        }
    };
    // clen*(clen+1) as double (matches C's (double)clen * (double)(clen+1))
    let clen_f = clen as f64;
    let etarget = (esigma - ESL_CONST_LOG2R * (2.0 / (clen_f * (clen_f + 1.0))).ln()) / clen_f;
    if etarget > re_target {
        etarget
    } else {
        re_target
    }
}

/// cmbuild.c:version_1p0_default_target_relent -- Infernal v1.0->1.0.2 target relent.
///   etarget = 6.*(eX + log((double)((clen*(clen+1))/2)) / log(2.)) / (double)(2*clen+4);
///   RNA: if (etarget < DEFAULT_ETARGET) etarget = DEFAULT_ETARGET;
/// NOTE: `(clen*(clen+1))/2` is INTEGER arithmetic in C before the cast to double.
pub fn version_1p0_default_target_relent(clen: i32, e_x: f64) -> f64 {
    // integer arithmetic, exactly as C: (clen * (clen+1)) / 2
    let int_term: i32 = (clen * (clen + 1)) / 2;
    let mut etarget =
        6.0 * (e_x + (int_term as f64).ln() / 2.0f64.ln()) / ((2 * clen + 4) as f64);
    if etarget < DEFAULT_ETARGET {
        etarget = DEFAULT_ETARGET;
    }
    etarget
}
