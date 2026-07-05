//! cm_nohmm — faithful port of C Infernal 1.1.5 cmsearch's `-g --nohmm` CM search.
//!
//! Implements the non-HMM, global (glocal) pipeline used by tRNAscan-SE Phase-II:
//!   pli_cyk_seq_filter()  — whole-window QDB-banded CYK filter → envelopes
//!   pli_final_stage()     — QDB-banded integer Inside per envelope → hits
//! mirroring cm_pipeline.c (do_edef==FALSE && do_fcyk==TRUE path). Global config:
//! the CM is NOT localized, so beginsc/endsc are all IMPOSSIBLE and model entry is
//! purely via the retained global root transitions cm.tsc[0] (the CMH_LOCAL_BEGIN
//! block in FastCYKScan/FastIInsideScan is skipped).
//!
//! DP scanners are faithful ports of FastCYKScan (float, ESL_MAX) and
//! FastIInsideScan (integer, ILogsum) from cm_dpsearch.c, unified via a generic
//! semiring so the recursion is written once.

use crate::cm::{CM, ALPHABET_SIZE_P};
use crate::constants::{B_ST, BEGL_S, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST};
use crate::cp9_faithful::{ilogsum, score_correction_null3, INFTY, INTSCALE};

const IMPOSSIBLE: f32 = -1.0e36;
#[inline]
fn not_impossible(x: f32) -> bool {
    x > IMPOSSIBLE + 1.0
}

/// C `Prob2Score(p, null)` (cm.c:4220): floor(0.5 + INTSCALE*log2(p/null)); -INFTY if p==0.
#[inline]
pub fn prob2score(p: f32, null: f32) -> i32 {
    if p == 0.0 {
        -INFTY
    } else {
        (0.5 + INTSCALE * (p as f64 / null as f64).log2()).floor() as i32
    }
}

/// C `Scorify(sc)` (cm.c:4243): sc==-INFTY ? IMPOSSIBLE : sc/INTSCALE.
#[inline]
fn scorify(sc: i32) -> f32 {
    if sc == -INFTY {
        IMPOSSIBLE
    } else {
        (sc as f64 / INTSCALE) as f32
    }
}

/// C sreLOG2 (infernal.h:153): x>0 ? log(x)*1.44269504 : IMPOSSIBLE.
#[inline]
fn sre_log2(x: f64) -> f64 {
    if x > 0.0 {
        x.ln() * 1.442_695_04
    } else {
        IMPOSSIBLE as f64
    }
}

// Emit modes (C Emitmode()).
const EMITNONE: i32 = 0;
const EMITLEFT: i32 = 1;
const EMITRIGHT: i32 = 2;
const EMITPAIR: i32 = 3;
#[inline]
fn emitmode(stt: i32) -> i32 {
    match stt {
        x if x == ML_ST || x == IL_ST => EMITLEFT,
        x if x == MR_ST || x == IR_ST => EMITRIGHT,
        x if x == MP_ST => EMITPAIR,
        _ => EMITNONE,
    }
}
#[inline]
fn state_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST => 2,
        x if x == ML_ST || x == MR_ST || x == IL_ST || x == IR_ST => 1,
        _ => 0,
    }
}
#[inline]
fn state_right_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST || x == MR_ST || x == IR_ST => 1,
        _ => 0,
    }
}

// ============================================================================
// Semiring: unifies CYK (f32, max) and Inside (i32, ilogsum).
// ============================================================================
trait Sr: Copy {
    fn zero() -> Self; // -inf sentinel (IMPOSSIBLE / -INFTY)
    fn from_zero() -> Self; // 0 bits (E-state base case / accumulator identity)
    fn comb(a: Self, b: Self) -> Self; // ESL_MAX / ILogsum
    fn addv(a: Self, b: Self) -> Self; // a + b
    fn scbits(a: Self) -> f32; // to reported bit score
}
impl Sr for f32 {
    #[inline]
    fn zero() -> Self {
        IMPOSSIBLE
    }
    #[inline]
    fn from_zero() -> Self {
        0.0
    }
    #[inline]
    fn comb(a: Self, b: Self) -> Self {
        if a >= b {
            a
        } else {
            b
        }
    }
    #[inline]
    fn addv(a: Self, b: Self) -> Self {
        a + b
    }
    #[inline]
    fn scbits(a: Self) -> f32 {
        a
    }
}
impl Sr for i32 {
    #[inline]
    fn zero() -> Self {
        -INFTY
    }
    #[inline]
    fn from_zero() -> Self {
        0
    }
    #[inline]
    fn comb(a: Self, b: Self) -> Self {
        ilogsum(a, b)
    }
    #[inline]
    fn addv(a: Self, b: Self) -> Self {
        a.wrapping_add(b)
    }
    #[inline]
    fn scbits(a: Self) -> f32 {
        scorify(a)
    }
}

/// A raw hit reported by a scanner (window-frame coords, null3-corrected score).
#[derive(Clone, Copy, Debug)]
pub struct RawHit {
    pub i: i32,
    pub j: i32,
    pub score: f32,
    pub bias: f32,
}

// QDB band indices (C SMX_*).
pub const SMX_QDB1_TIGHT: usize = 1;
pub const SMX_QDB2_LOOSE: usize = 2;

// ============================================================================
// Global configuration: build float + integer log-odds without localizing.
// ============================================================================
/// C `cm_Configure` with `-g` (no CM_CONFIG_LOCAL): the model stays global.
/// Builds cm.tsc/esc/oesc (float) and cm.itsc/ioesc/ibeginsc/iendsc/iel_selfsc
/// (integer). begin[]/end[] are the file 0's, so beginsc/endsc = IMPOSSIBLE and
/// ibeginsc/iendsc = -INFTY (model entry via global cm.tsc[0] only).
pub fn cm_configure_scores_global(cm: &mut CM) {
    // clamp el_selfsc (cm.c: el_selfsc*W must be finite)
    if (cm.el_selfsc * cm.w as f32) < IMPOSSIBLE {
        cm.el_selfsc = IMPOSSIBLE / (cm.w as f32 + 1.0);
    }
    // float scores (tsc/esc/oesc/beginsc/endsc); begin[]/end[] stay 0 => IMPOSSIBLE
    crate::cp9_faithful::cm_logoddsify(cm);
    build_integer_scores(cm);
}

/// Build cm.itsc / cm.ioesc / cm.ibeginsc / cm.iendsc / cm.iel_selfsc, mirroring
/// C CMLogoddsify() + ICalcOptimizedEmitScores().
fn build_integer_scores(cm: &mut CM) {
    let m = cm.m as usize;
    let k = crate::cm::ALPHABET_SIZE; // 4
    let kp = ALPHABET_SIZE_P; // 18
    cm.itsc = vec![Vec::new(); m];
    cm.ioesc = vec![Vec::new(); m];
    cm.ibeginsc = vec![-INFTY; m];
    cm.iendsc = vec![-INFTY; m];
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        if stt != B_ST && stt != E_ST {
            let cnum = cm.cnum[v] as usize;
            let mut itsc = vec![0i32; cnum];
            for x in 0..cnum {
                itsc[x] = prob2score(cm.t[v][x], 1.0);
            }
            cm.itsc[v] = itsc;
        }
        // integer optimized emission scores. Mirrors C ICalcOptimizedEmitScores:
        // canonical from the K*K/K iesc; degenerate (K+1..Kp-1) marginalized
        // (pairs = truncated fraction-weighted sums via iFastPairScore*, singlets
        // = round-half-away IAvgScore); gap/missing/unfilled stay -INFTY.
        if stt == MP_ST {
            // canonical iesc (== C cm->iesc[v], K*K), then Kp*Kp ioesc.
            let mut iesc = vec![0i32; k * k];
            for a in 0..k {
                for b in 0..k {
                    iesc[a * k + b] = prob2score(cm.e[v][a * k + b], cm.null[a] * cm.null[b]);
                }
            }
            let mut ioesc = vec![-INFTY; kp * kp];
            for a in 0..k {
                for b in 0..k {
                    ioesc[a * kp + b] = iesc[a * k + b];
                }
            }
            // a degenerate, b canonical: iFastPairScoreLeftOnlyDegenerate
            for a in (k + 1)..(kp - 1) {
                for b in 0..k {
                    let mut sc = 0.0f32;
                    for l in 0..k {
                        sc += iesc[l * k + b] as f32 * crate::cm::abc_fcount_frac(a, l);
                    }
                    ioesc[a * kp + b] = sc as i32;
                }
            }
            // a canonical, b degenerate: iFastPairScoreRightOnlyDegenerate
            for a in 0..k {
                for b in (k + 1)..(kp - 1) {
                    let mut sc = 0.0f32;
                    for r in 0..k {
                        sc += iesc[a * k + r] as f32 * crate::cm::abc_fcount_frac(b, r);
                    }
                    ioesc[a * kp + b] = sc as i32;
                }
            }
            // both degenerate: iFastPairScoreBothDegenerate
            for a in (k + 1)..(kp - 1) {
                for b in (k + 1)..(kp - 1) {
                    let mut sc = 0.0f32;
                    for l in 0..k {
                        for r in 0..k {
                            sc += iesc[l * k + r] as f32
                                * crate::cm::abc_fcount_frac(a, l)
                                * crate::cm::abc_fcount_frac(b, r);
                        }
                    }
                    ioesc[a * kp + b] = sc as i32;
                }
            }
            cm.ioesc[v] = ioesc;
        } else if stt == ML_ST || stt == MR_ST || stt == IL_ST || stt == IR_ST {
            // canonical iesc (== C cm->iesc[v], K), then Kp ioesc.
            let mut iesc = vec![0i32; k];
            for a in 0..k {
                iesc[a] = prob2score(cm.e[v][a], cm.null[a]);
            }
            let mut ioesc = vec![-INFTY; kp];
            for a in 0..k {
                ioesc[a] = iesc[a];
            }
            for a in (k + 1)..(kp - 1) {
                ioesc[a] = crate::cm::abc_iavg_score(a, &iesc);
            }
            cm.ioesc[v] = ioesc;
        }
        cm.ibeginsc[v] = prob2score(cm.begin[v], 1.0);
        cm.iendsc[v] = prob2score(cm.end[v], 1.0);
    }
    cm.iel_selfsc = prob2score(2f32.powf(cm.el_selfsc), 1.0);
}

/// Compute QDB bands (dmin1/dmax1 @ beta1, dmin2/dmax2 @ beta2) into the CM, using
/// the legacy density engine (faithful port of BandCalculationEngine). W stays as
/// read from the file (computed at the same beta_W).
pub fn cm_calc_qdb_bands(cm: &mut CM, beta1: f64, beta2: f64, beta_w: f64) -> Result<(), String> {
    use crate::legacy::cm_qdband::{calculate_query_dependent_bands, QdbInfo};
    let mut qi = QdbInfo::new(cm.m as usize, cm.clen);
    qi.beta1 = beta1;
    qi.beta2 = beta2;
    let res = calculate_query_dependent_bands(cm, Some(qi), beta_w, false, false)?;
    let qi = res.qdbinfo.ok_or_else(|| "qdbinfo missing".to_string())?;
    cm.dmin1 = qi.dmin1;
    cm.dmax1 = qi.dmax1;
    cm.dmin2 = qi.dmin2;
    cm.dmax2 = qi.dmax2;
    cm.qdb_beta1 = beta1;
    cm.qdb_beta2 = beta2;
    Ok(())
}

// ============================================================================
// Per-type score tables handed to the generic scanner.
// ============================================================================
struct Tables<'a, S: Sr> {
    tsc: &'a [Vec<S>],
    oesc: &'a [Vec<S>],
    endsc: Vec<S>,     // per-state end score (S)
    el_self: S,        // EL self-transition score (S)
}

// ============================================================================
// Generic scanner: faithful port of FastCYKScan / FastIInsideScan (global mode).
// Returns greedily-resolved, overlap-pruned hits in [i0..j0] window coords.
// ============================================================================
#[allow(clippy::too_many_arguments)]
fn generic_scan<S: Sr>(
    cm: &CM,
    tab: &Tables<S>,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    cutoff: f32,
    do_null3: bool,
    qdbidx: usize,
    nonbanded: bool,
) -> Vec<RawHit> {
    let m = cm.m as usize;
    let kp = ALPHABET_SIZE_P;
    let smx_w = cm.w; // scan matrix W (== cm.W)
    let l = j0 - i0 + 1;
    let mut w = smx_w;
    if w > l {
        w = l;
    }
    let w = w as usize;

    // band vectors for this qdbidx. `nonbanded` (C SMX_NOQDB, the `-g --max` path)
    // replaces QDBs with the full d-range: dmin=0, dmax=W for every state, so the
    // per-state dn/dx below collapse to (MP?2:1)..min(j,W) — nothing is pruned.
    let full_dmin: Vec<i32>;
    let full_dmax: Vec<i32>;
    let (dmin, dmax): (&[i32], &[i32]) = if nonbanded {
        full_dmin = vec![0i32; m];
        full_dmax = vec![smx_w; m];
        (&full_dmin, &full_dmax)
    } else if qdbidx == SMX_QDB1_TIGHT {
        (&cm.dmin1, &cm.dmax1)
    } else {
        (&cm.dmin2, &cm.dmax2)
    };

    // Precompute per-state dn (j-independent) and dxcap (=min(dmax[v],W)); dx=min(jrow,dxcap).
    let mut dn_v = vec![0i32; m];
    let mut dxcap_v = vec![0i32; m];
    for v in 0..m {
        let base = if cm.sttype[v] as i32 == MP_ST { 2 } else { 1 };
        let mut dn = dmin[v].max(base);
        if dn > smx_w {
            dn = smx_w;
        }
        dn_v[v] = dn;
        dxcap_v[v] = dmax[v].min(smx_w);
    }

    // BEGL_S compact indexing
    let mut begl_idx = vec![usize::MAX; m];
    let mut nbegl = 0usize;
    for v in 0..m {
        if cm.stid[v] as i32 == BEGL_S {
            begl_idx[v] = nbegl;
            nbegl += 1;
        }
    }

    // init scores: init_scAA[v][d] = endsc[v]==IMPOSSIBLE ? zero : el_self*d + endsc[v]
    // (global: endsc all IMPOSSIBLE => all zero)
    // init_scAA[v][d] = endsc[v] valid ? el_self*d + endsc[v] : zero.
    // Precompute el_scA[d] = el_self*d once (repeated add, exact like C's multiply-by-int).
    let mut el_sca = vec![S::from_zero(); w + 1];
    for d in 1..=w {
        el_sca[d] = S::addv(el_sca[d - 1], tab.el_self);
    }
    let mut init_sc = vec![vec![S::zero(); w + 1]; m];
    for v in 0..m {
        let es = tab.endsc[v];
        if not_impossible(S::scbits(es)) {
            for d in 0..=w {
                init_sc[v][d] = S::addv(el_sca[d], es);
            }
        }
    }

    // alpha[2][M][W+1] (non-BEGL_S); alpha_begl[W+1][nbegl][W+1]
    let mut alpha = vec![vec![vec![S::zero(); w + 1]; m]; 2];
    let mut alpha_begl = vec![vec![vec![S::zero(); w + 1]; nbegl]; w + 1];

    // ---- initialize d=0 base cases (cm_scan_mx_Initialize) ----
    for v in (0..m).rev() {
        if cm.stid[v] as i32 != BEGL_S {
            let stt = cm.sttype[v] as i32;
            if stt == E_ST {
                // E: alpha[*][v][0] = 0
                alpha[0][v][0] = S::from_zero();
                alpha[1][v][0] = S::from_zero();
            } else if stt == S_ST || stt == D_ST {
                let y = cm.cfirst[v] as usize;
                let mut a0 = tab.endsc[v];
                for yo in 0..cm.cnum[v] as usize {
                    a0 = S::comb(a0, S::addv(alpha[0][y + yo][0], tab.tsc[v][yo]));
                }
                a0 = S::comb(a0, S::zero());
                alpha[0][v][0] = a0;
                alpha[1][v][0] = a0;
            } else if stt == B_ST {
                let wl = cm.cfirst[v] as usize; // BEGL_S
                let y = cm.cnum[v] as usize; // BEGR_S
                let a0 = S::addv(alpha_begl[0][begl_idx[wl]][0], alpha[0][y][0]);
                alpha[0][v][0] = a0;
                alpha[1][v][0] = a0;
            } else {
                // emitters: d=0 stays zero() (IMPOSSIBLE)
                alpha[1][v][0] = alpha[0][v][0];
            }
        } else {
            // BEGL_S
            let bi = begl_idx[v];
            let y = cm.cfirst[v] as usize;
            let mut a0 = tab.endsc[v];
            for yo in 0..cm.cnum[v] as usize {
                a0 = S::comb(a0, S::addv(alpha[0][y + yo][0], tab.tsc[v][yo]));
            }
            a0 = S::comb(a0, S::zero());
            for j in 0..=w {
                alpha_begl[j][bi][0] = a0;
            }
        }
    }

    // ---- null3 act vector ----
    let mut act: Vec<[f64; 4]> = if do_null3 {
        vec![[0.0; 4]; w + 1]
    } else {
        Vec::new()
    };

    let mut jp_wa = vec![0usize; w + 1];
    let mut sc_v = vec![S::zero(); w + 1];
    let mut bestsc = vec![IMPOSSIBLE; w + 1];
    let mut bestr = vec![-1i32; w + 1];

    let mut tmp_hits: Vec<RawHit> = Vec::new();

    // ---- main scan over end positions j ----
    for j in i0..=j0 {
        let jp_g = (j - i0 + 1) as usize;
        let cur = (j & 1) as usize;
        let prv = ((j - 1) & 1) as usize;
        let jrow = jp_g.min(w);
        for d in 0..=w {
            jp_wa[d] = ((j - d as i32).rem_euclid(w as i32 + 1)) as usize;
        }
        if do_null3 {
            let src = (jp_g - 1) % (w + 1);
            let dst = jp_g % (w + 1);
            act[dst] = act[src];
            let dj = dsq[j as usize];
            if (dj as usize) < 4 {
                act[dst][dj as usize] += 1.0;
            }
        }

        for v in (1..m).rev() {
            let stt = cm.sttype[v] as i32;
            if stt == E_ST {
                continue;
            }
            let sd = state_delta(stt);
            let em = emitmode(stt);
            let jp_v = if cm.stid[v] as i32 == BEGL_S {
                (j.rem_euclid(w as i32 + 1)) as usize
            } else {
                cur
            };
            let jp_y = if state_right_delta(stt) > 0 { prv } else { cur };
            // dn is clamped to W only (j-independent); dx carries the min(j, ...).
            // When jrow < dn the d-loops are naturally empty (dn > dx).
            let dn = dn_v[v];
            let dx = (jrow as i32).min(dxcap_v[v]);
            let esc_v: &[S] = &tab.oesc[v];
            let esc_j = if stt == IR_ST || stt == MR_ST {
                esc_v[dsq[j as usize] as usize]
            } else {
                S::zero()
            };

            if stt == B_ST {
                let wl = cm.cfirst[v] as usize; // BEGL_S
                let y = cm.cnum[v] as usize; // BEGR_S
                let bi = begl_idx[wl];
                for d in dn..=dx {
                    let dn_y = dmin[y].min(smx_w);
                    let dx_y = dmax[y].min(smx_w);
                    let dn_w = dmin[wl].min(smx_w);
                    let dx_w = dmax[wl].min(smx_w);
                    let kmin = 0.max(dn_y.max(d - dx_w));
                    let kmax = dx_y.min(d - dn_w);
                    let mut sc = init_sc[v][(d - sd) as usize];
                    let mut kk = kmin;
                    while kk <= kmax {
                        let left = alpha_begl[jp_wa[kk as usize]][bi][(d - kk) as usize];
                        let right = alpha[jp_y][y][kk as usize];
                        sc = S::comb(sc, S::addv(left, right));
                        kk += 1;
                    }
                    alpha[jp_v][v][d as usize] = sc;
                }
            } else if cm.stid[v] as i32 == BEGL_S {
                let y = cm.cfirst[v] as usize;
                let bi = begl_idx[v];
                for d in dn..=dx {
                    let mut sc = init_sc[v][(d - sd) as usize];
                    for yo in 0..cm.cnum[v] as usize {
                        sc = S::comb(sc, S::addv(alpha[jp_y][y + yo][(d - sd) as usize], tab.tsc[v][yo]));
                    }
                    alpha_begl[jp_v][bi][d as usize] = sc;
                }
            } else if stt == IL_ST || stt == IR_ST {
                let y = cm.cfirst[v] as usize;
                let tsc_v = &tab.tsc[v];
                let mut i = j - dn + 1;
                let mut dp_y = dn - sd;
                for d in dn..=dx {
                    let mut sc = init_sc[v][dp_y as usize];
                    for yo in 0..cm.cnum[v] as usize {
                        sc = S::comb(sc, S::addv(alpha[jp_y][y + yo][dp_y as usize], tsc_v[yo]));
                    }
                    match em {
                        EMITLEFT => {
                            sc = S::addv(sc, esc_v[dsq[i as usize] as usize]);
                            i -= 1;
                        }
                        EMITRIGHT => {
                            sc = S::addv(sc, esc_j);
                        }
                        _ => {}
                    }
                    alpha[jp_v][v][d as usize] = sc;
                    dp_y += 1;
                }
            } else {
                let y = cm.cfirst[v] as usize;
                let tsc_v = &tab.tsc[v];
                let mut dp_y = dn - sd;
                for d in dn..=dx {
                    let mut s = init_sc[v][dp_y as usize];
                    for yo in 0..cm.cnum[v] as usize {
                        s = S::comb(s, S::addv(alpha[jp_y][y + yo][dp_y as usize], tsc_v[yo]));
                    }
                    sc_v[d as usize] = s;
                    dp_y += 1;
                }
                match em {
                    EMITLEFT => {
                        let mut i = j - dn + 1;
                        for d in dn..=dx {
                            alpha[jp_v][v][d as usize] =
                                S::addv(sc_v[d as usize], esc_v[dsq[i as usize] as usize]);
                            i -= 1;
                        }
                    }
                    EMITNONE => {
                        for d in dn..=dx {
                            alpha[jp_v][v][d as usize] = sc_v[d as usize];
                        }
                    }
                    EMITRIGHT => {
                        for d in dn..=dx {
                            alpha[jp_v][v][d as usize] = S::addv(sc_v[d as usize], esc_j);
                        }
                    }
                    EMITPAIR => {
                        let mut i = j - dn + 1;
                        for d in dn..=dx {
                            let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                            alpha[jp_v][v][d as usize] = S::addv(sc_v[d as usize], esc_v[idx]);
                            i -= 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        // ---- ROOT_S (v=0), global mode: plain root transitions, no local begins ----
        let dn0 = dn_v[0];
        let dx0 = (jrow as i32).min(dxcap_v[0]);
        let tsc0 = &tab.tsc[0];
        let y0 = cm.cfirst[0] as usize;
        for d in dn0..=dx0 {
            bestr[d as usize] = 0;
            let mut a = S::comb(S::zero(), S::addv(alpha[cur][y0][d as usize], tsc0[0]));
            for yo in 1..cm.cnum[0] as usize {
                a = S::comb(a, S::addv(alpha[cur][y0 + yo][d as usize], tsc0[yo]));
            }
            alpha[cur][0][d as usize] = a;
        }
        for d in dn0..=dx0 {
            bestsc[d as usize] = S::scbits(alpha[cur][0][d as usize]);
        }

        report_hits_greedily(
            cm, j, dn0, dx0, &bestsc, &bestr, w, if do_null3 { Some(&act) } else { None },
            i0, cutoff, &mut tmp_hits,
        );
    }

    // ---- greedy overlap removal (SortForOverlapRemoval + RemoveOrMarkOverlaps) ----
    let raw: Vec<(i32, i32, f32, f32)> =
        tmp_hits.iter().map(|h| (h.i, h.j, h.score, h.bias)).collect();
    let kept = crate::cp9_faithful::remove_overlaps_greedy(raw);
    kept.into_iter().map(|(i, j, s, b)| RawHit { i, j, score: s, bias: b }).collect()
}

// ============================================================================
// report_hits_greedily — C ReportHitsGreedily (cm_mx.c:7555), std (non-trunc).
// ============================================================================
#[allow(clippy::too_many_arguments)]
fn report_hits_greedily(
    cm: &CM,
    j: i32,
    dmin: i32,
    dmax: i32,
    bestsc: &[f32],
    bestr: &[i32],
    w: usize,
    act: Option<&Vec<[f64; 4]>>,
    i0: i32,
    cutoff: f32,
    out: &mut Vec<RawHit>,
) {
    if dmin > dmax {
        return;
    }
    let mut max_reported = IMPOSSIBLE;
    let dlo = dmin.max(1);
    for d in dlo..=dmax {
        let i = j - d + 1;
        let mut hit_sc = bestsc[d as usize];
        let mut bias = 0.0f32;
        if hit_sc > max_reported && hit_sc >= cutoff && not_impossible(hit_sc) {
            let mut do_report = true;
            if let Some(act) = act {
                let ip = (i - i0 + 1) as usize;
                let jp = (j - i0 + 1) as usize;
                let mut comp = [0.0f32; 4];
                for a in 0..4 {
                    comp[a] = (act[jp % (w + 1)][a] - act[(ip - 1) % (w + 1)][a]) as f32;
                }
                esl_vec_fnorm(&mut comp);
                let corr = score_correction_null3(&cm.null, &comp, d, cm.n3_omega as f32);
                hit_sc -= corr;
                bias = corr;
                do_report = hit_sc > max_reported && hit_sc >= cutoff;
            }
            if do_report {
                out.push(RawHit { i, j, score: hit_sc, bias });
                max_reported = hit_sc;
            }
        }
    }
}

#[inline]
fn esl_vec_fnorm(v: &mut [f32]) {
    let sum: f32 = v.iter().sum();
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    } else {
        let n = v.len() as f32;
        for x in v.iter_mut() {
            *x = 1.0 / n;
        }
    }
}

// ============================================================================
// Public pipeline: CYK seq filter + final Inside stage.
// ============================================================================

fn build_float_tables(cm: &CM) -> Tables<'_, f32> {
    Tables {
        tsc: &cm.tsc,
        oesc: &cm.oesc,
        endsc: cm.endsc.clone(),
        el_self: cm.el_selfsc,
    }
}
fn build_int_tables(cm: &CM) -> Tables<'_, i32> {
    Tables {
        tsc: &cm.itsc,
        oesc: &cm.ioesc,
        endsc: cm.iendsc.clone(),
        el_self: cm.iel_selfsc,
    }
}

/// C pli_cyk_seq_filter: whole-window QDB-CYK filter → padded/merged envelopes.
/// dsq is window-frame ([255, res.., 255]); n = window length; returns (es,ee) in
/// window coords.
pub fn cyk_seq_filter(cm: &CM, dsq: &[u8], n: i32, cutoff: f32, do_null3: bool) -> Vec<(i32, i32)> {
    let tab = build_float_tables(cm);
    let hits = generic_scan::<f32>(cm, &tab, dsq, 1, n, cutoff, do_null3, SMX_QDB1_TIGHT, false);
    // pad each hit to a window ±(W-1) and greedily merge overlapping windows.
    // C sorts by end position (SortByPosition) then merges.
    let mut padded: Vec<(i32, i32)> = hits
        .iter()
        .map(|h| {
            let iwin = 1.max(h.j - (cm.w - 1));
            let jwin = n.min(h.i + (cm.w - 1));
            (iwin, jwin)
        })
        .collect();
    // sort by jwin ascending (C sorts hits by increasing end point j before merge)
    padded.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut out: Vec<(i32, i32)> = Vec::new();
    let mut h = 0;
    while h < padded.len() {
        let iwin = padded[h].0;
        let mut jwin = padded[h].1;
        // merge subsequent windows whose next_iwin <= jwin
        while h + 1 < padded.len() {
            let next_iwin = padded[h + 1].0;
            if next_iwin <= jwin {
                h += 1;
                jwin = jwin.max(padded[h].1);
            } else {
                break;
            }
        }
        out.push((iwin, jwin));
        h += 1;
    }
    out
}

/// mdl from/to (C ad->cfrom_emit / cto_emit) for a hit spanning window residues
/// `hi..hj`, from a global CYK alignment of the subsequence to the CM followed by
/// ParsetreeToCMBounds. `cm` must be the global-config CM. This is the standalone
/// API; the pipeline path uses [`crate::cm_alidisplay::cm_alidisplay_create`]
/// directly (which also returns the full alignment display).
pub fn cyk_align_cmbounds(cm: &CM, wdsq: &[u8], hi: i32, hj: i32) -> (i32, i32) {
    let lp = hj - hi + 1;
    let subdsq = &wdsq[(hi as usize - 1)..];
    let cmcons = crate::cm_alidisplay::create_cm_consensus(cm);
    let (tr, _sc) = crate::cm_alidisplay::cyk_align_global(cm, subdsq, lp);
    let ad = crate::cm_alidisplay::cm_alidisplay_create(cm, &cmcons, &tr, subdsq);
    (ad.cfrom_emit, ad.cto_emit)
}

/// C pli_final_stage: QDB integer Inside on each envelope → hits (window coords).
/// Returns (i, j, score, bias). Score is null3-corrected reported bit score.
pub fn final_stage_inside(
    cm: &CM,
    dsq: &[u8],
    envs: &[(i32, i32)],
    cutoff: f32,
    do_null3: bool,
) -> Vec<RawHit> {
    final_stage_inside_opt(cm, dsq, envs, cutoff, do_null3, false)
}

/// As [`final_stage_inside`], but `nonbanded=true` selects the full d-range
/// (C SMX_NOQDB) used by the `-g --max` final Inside stage.
pub fn final_stage_inside_opt(
    cm: &CM,
    dsq: &[u8],
    envs: &[(i32, i32)],
    cutoff: f32,
    do_null3: bool,
    nonbanded: bool,
) -> Vec<RawHit> {
    let tab = build_int_tables(cm);
    let mut out = Vec::new();
    for &(es, ee) in envs {
        let hits =
            generic_scan::<i32>(cm, &tab, dsq, es, ee, cutoff, do_null3, SMX_QDB2_LOOSE, nonbanded);
        out.extend(hits);
    }
    out
}
