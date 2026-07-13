//! cm_dpalign — faithful port of Infernal 1.1.5 `cm_dpalign.c` NON-banded,
//! non-D&C CM alignment DP (the core of `cmalign --nonbanded`).
//!
//! Ported functions (C `cm_dpalign.c`):
//!   cm_alignT()                    (cm_dpalign.c:117)  -> cm_align_t()
//!   cm_Align()                     (cm_dpalign.c:644)  -> cm_align()
//!   cm_CYKInsideAlign()            (cm_dpalign.c:846)  -> cm_cyk_inside_align()
//!   cm_InsideAlign()               (cm_dpalign.c:1552) -> cm_inside_align()
//!   cm_OptAccAlign()               (cm_dpalign.c:2205) -> cm_optacc_align()
//!   cm_OutsideAlign()              (cm_dpalign.c:4013) -> cm_outside_align()
//!   cm_Posterior()                 (cm_dpalign.c:4819) -> cm_posterior()
//!   cm_EmitterPosterior()          (cm_dpalign.c:5044) -> cm_emitter_posterior()
//!   cm_PostCode()                  (cm_dpalign.c:5450) -> cm_postcode()
//!   cm_InitializeOptAccShadowDZero (cm_dpalign.c:5646) -> cm_init_optacc_shadow_dzero()
//! Plus the dense (non-banded) [v][j][d] matrix types from cm_mx.c (CM_MX,
//! CM_SHADOW_MX, CM_EMIT_MX) and a local port of Parsetrees2Alignment()
//! (cm_parsetree.c:846) restricted to the cmalign call (do_full=TRUE,
//! do_matchonly=FALSE, allow_trunc=FALSE, all J-mode / non-truncated parses).
//!
//! All local-mode machinery (local begins/ends, EL deck) is ported so the DP is
//! correct whether the CM is configured global (`-g`) or local (cmalign default).

use crate::cm::{CM, ALPHABET_SIZE_P};
use crate::constants::{
    B_ST, BEGL_S, BEGR_S, D_ST, E_ST, EL_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST,
    MATP_ND, MATL_ND, MATR_ND, BIF_ND, MATP_D, MATL_D, MATR_D,
};
use crate::cp9::{flogsum, CP9Bands};
use crate::parsetree::Parsetree;
use crate::cm_emitmap::create_emit_map;
use crate::cm_alidisplay::{create_cm_consensus, CmConsensus};
use crate::easel::alphabet::EslAlphabet;
use crate::easel::msa::EslMsa;
use crate::easel::random::EslRandom;

// C infernal.h:148 / :151
pub const IMPOSSIBLE: f32 = -1.0e36;
#[inline]
pub fn not_impossible(x: f32) -> bool {
    x > -9.999e35
}
// C infernal.h:323-324. yshadow sentinels (stored as i32 here).
const USED_LOCAL_BEGIN: i32 = 101;
const USED_EL: i32 = 102;

// CM flag bits (must match cp9 / cm_search).
const CMH_LOCAL_BEGIN: u32 = 1 << 10;
const CMH_LOCAL_END: u32 = 1 << 11;

/// C sreEXP2(x) = exp(x*0.69314718) (infernal.h:154), evaluated in double.
#[inline]
fn sre_exp2(x: f32) -> f64 {
    ((x as f64) * 0.69314718f64).exp()
}

// C StateDelta / StateLeftDelta / StateRightDelta (cm.c).
#[inline]
fn state_delta(stt: i32) -> i32 {
    if stt == MP_ST { 2 } else if stt == ML_ST || stt == MR_ST || stt == IL_ST || stt == IR_ST { 1 } else { 0 }
}
#[inline]
fn state_left_delta(stt: i32) -> i32 {
    if stt == MP_ST || stt == ML_ST || stt == IL_ST { 1 } else { 0 }
}
#[inline]
fn state_right_delta(stt: i32) -> i32 {
    if stt == MP_ST || stt == MR_ST || stt == IR_ST { 1 } else { 0 }
}

// ============================================================================
// Dense matrix types (cm_mx.c). Layout: dp[v] is a flat (L+1)*(L+1) buffer,
// cell [j][d] at index j*(L+1)+d. Decks 0..M are model states; deck M is EL.
// ============================================================================

/// C CM_MX (float DP cube). `dp[v][j*stride+d]`.
pub struct CmMx {
    pub dp: Vec<Vec<f32>>, // 0..=M (M+1 decks)
    pub stride: usize,     // L+1
    pub l: usize,
}
impl CmMx {
    fn new(m: usize, l: usize) -> Self {
        let stride = l + 1;
        CmMx { dp: vec![vec![IMPOSSIBLE; stride * stride]; m + 1], stride, l }
    }
}

/// C CM_SHADOW_MX. yshadow for non-B states, kshadow for B states.
pub struct CmShadowMx {
    pub yshadow: Vec<Vec<i32>>, // 0..M-1
    pub kshadow: Vec<Vec<i32>>, // 0..M-1 (only B meaningful)
    pub stride: usize,
}

/// C CM_EMIT_MX. l_pp[v][i], r_pp[v][j] (1..L), plus per-residue sum[i].
pub struct CmEmitMx {
    pub l_pp: Vec<Option<Vec<f32>>>, // 0..=M
    pub r_pp: Vec<Option<Vec<f32>>>, // 0..=M
    pub sum: Vec<f32>,               // 0..L
}

// ============================================================================
// cm_CYKInsideAlign (cm_dpalign.c:846)
// ============================================================================
/// Returns (mx, shadow, b, sc). sc = alpha[0][L][L].
fn cm_cyk_inside_align(cm: &CM, dsq: &[u8], l: i32) -> (CmMx, CmShadowMx, i32, f32) {
    let m = cm.m as usize;
    let ll = l as usize;
    let mut mx = CmMx::new(m, ll);
    let stride = mx.stride;
    let ncells = stride * stride;
    let mut yshadow: Vec<Vec<i32>> = vec![vec![USED_EL; ncells]; m];
    let mut kshadow: Vec<Vec<i32>> = vec![vec![USED_EL; ncells]; m];

    let kp = ALPHABET_SIZE_P;
    let mut b: i32 = -1;
    let mut bsc: f32 = IMPOSSIBLE;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    // el_scA[d] = el_selfsc * d  (cm_dpalign.c:884)
    let mut el_sca = vec![0.0f32; ll + 1];
    for d in 0..=ll {
        el_sca[d] = cm.el_selfsc * d as f32;
    }
    // EL deck (cm_dpalign.c:887)
    if local_end {
        for j in 0..=ll {
            for d in 0..=j {
                mx.dp[m][j * stride + d] = el_sca[d];
            }
        }
    }

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let endsc_v = cm.endsc[v];

        // build current deck `a` + shadow `ys`/`ks` locally, then store.
        let mut a = vec![IMPOSSIBLE; ncells];
        let mut ys = vec![USED_EL; ncells];
        let mut ks = vec![USED_EL; ncells];

        // re-initialize J deck if we can do a local end from v (cm_dpalign.c:901)
        if not_impossible(endsc_v) {
            for j in 0..=(l) {
                for d in sd..=j {
                    a[(j * stride as i32 + d) as usize] = el_sca[(d - sd) as usize] + endsc_v;
                }
            }
        }

        if stt == E_ST {
            for j in 0..=ll {
                a[j * stride] = 0.0;
            }
        } else if stt == IL_ST {
            // for j { for d { for y } }  (self-transit) (cm_dpalign.c:916)
            for j in sdr..=l {
                let j_sdr = j - sdr;
                for d in sd..=j {
                    let d_sd = d - sd;
                    let i = j - d + 1;
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        let child = if y == v {
                            a[(j_sdr * stride as i32 + d_sd) as usize]
                        } else {
                            mx.dp[y][(j_sdr * stride as i32 + d_sd) as usize]
                        };
                        let sc = child + tsc_v[yo];
                        if sc > a[idx] {
                            a[idx] = sc;
                            ys[idx] = yo as i32;
                        }
                    }
                    a[idx] += esc_v[dsq[i as usize] as usize];
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else if stt == IR_ST {
            for j in sdr..=l {
                let j_sdr = j - sdr;
                for d in sd..=j {
                    let d_sd = d - sd;
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        let child = if y == v {
                            a[(j_sdr * stride as i32 + d_sd) as usize]
                        } else {
                            mx.dp[y][(j_sdr * stride as i32 + d_sd) as usize]
                        };
                        let sc = child + tsc_v[yo];
                        if sc > a[idx] {
                            a[idx] = sc;
                            ys[idx] = yo as i32;
                        }
                    }
                    a[idx] += esc_v[dsq[j as usize] as usize];
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else if stt != B_ST {
            // ML, MP, MR, D, S : no self-transit (cm_dpalign.c:960)
            for j in sdr..=l {
                let j_sdr = j - sdr;
                for d in sd..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        let sc = mx.dp[y][(j_sdr * stride as i32 + (d - sd)) as usize] + tsc_v[yo];
                        if sc > a[idx] {
                            a[idx] = sc;
                            ys[idx] = yo as i32;
                        }
                    }
                }
            }
            // add emission
            match stt {
                x if x == ML_ST => {
                    for j in 0..=l {
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            a[idx] += esc_v[dsq[(j - d + 1) as usize] as usize];
                        }
                    }
                }
                x if x == MR_ST => {
                    for j in 0..=l {
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            a[idx] += esc_v[dsq[j as usize] as usize];
                        }
                    }
                }
                x if x == MP_ST => {
                    for j in 0..=l {
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            let li = dsq[(j - d + 1) as usize] as usize;
                            let ri = dsq[j as usize] as usize;
                            a[idx] += esc_v[li * kp + ri];
                        }
                    }
                }
                _ => {}
            }
            // floor
            for j in 0..=ll {
                for d in 0..=j {
                    let idx = j * stride + d;
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else {
            // B_st (cm_dpalign.c:1011)
            let y = cfirst; // left  (BEGL_S)
            let z = cm.cnum[v] as usize; // right (BEGR_S)
            for j in 0..=l {
                for d in 0..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    for k in 0..=d {
                        let sc = mx.dp[y][((j - k) * stride as i32 + (d - k)) as usize]
                            + mx.dp[z][(j * stride as i32 + k) as usize];
                        if sc > a[idx] {
                            a[idx] = sc;
                            ks[idx] = k;
                        }
                    }
                }
            }
        }

        // local begins (cm_dpalign.c:1028)
        let root_idx = (l * stride as i32 + l) as usize;
        if local_begin && not_impossible(cm.beginsc[v]) && (a[root_idx] + cm.beginsc[v] > bsc) {
            b = v as i32;
            bsc = a[root_idx] + cm.beginsc[v];
        }

        mx.dp[v] = a;
        yshadow[v] = ys;
        kshadow[v] = ks;
    }

    // fold best local begin into root (cm_dpalign.c:1040)
    let root_idx = (l * stride as i32 + l) as usize;
    if bsc > mx.dp[0][root_idx] {
        mx.dp[0][root_idx] = bsc;
        yshadow[0][root_idx] = USED_LOCAL_BEGIN;
    }
    let sc = mx.dp[0][root_idx];
    let sh = CmShadowMx { yshadow, kshadow, stride };
    (mx, sh, b, sc)
}

// ============================================================================
// cm_InsideAlign (cm_dpalign.c:1552) — FLogsum, no shadow.
// ============================================================================
fn cm_inside_align(cm: &CM, dsq: &[u8], l: i32) -> (CmMx, f32) {
    let m = cm.m as usize;
    let ll = l as usize;
    let mut mx = CmMx::new(m, ll);
    let stride = mx.stride;
    let ncells = stride * stride;
    let kp = ALPHABET_SIZE_P;
    let mut bsc = IMPOSSIBLE;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let mut el_sca = vec![0.0f32; ll + 1];
    for d in 0..=ll {
        el_sca[d] = cm.el_selfsc * d as f32;
    }
    if local_end {
        for j in 0..=ll {
            for d in 0..=j {
                mx.dp[m][j * stride + d] = el_sca[d];
            }
        }
    }

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let endsc_v = cm.endsc[v];

        let mut a = vec![IMPOSSIBLE; ncells];
        if not_impossible(endsc_v) {
            for j in 0..=l {
                for d in sd..=j {
                    a[(j * stride as i32 + d) as usize] = el_sca[(d - sd) as usize] + endsc_v;
                }
            }
        }

        if stt == E_ST {
            for j in 0..=ll {
                a[j * stride] = 0.0;
            }
        } else if stt == IL_ST {
            for j in sdr..=l {
                let j_sdr = j - sdr;
                for d in sd..=j {
                    let d_sd = d - sd;
                    let i = j - d + 1;
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        let child = if y == v {
                            a[(j_sdr * stride as i32 + d_sd) as usize]
                        } else {
                            mx.dp[y][(j_sdr * stride as i32 + d_sd) as usize]
                        };
                        a[idx] = flogsum(a[idx], child + tsc_v[yo]);
                    }
                    a[idx] += esc_v[dsq[i as usize] as usize];
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else if stt == IR_ST {
            for j in sdr..=l {
                let j_sdr = j - sdr;
                for d in sd..=j {
                    let d_sd = d - sd;
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        let child = if y == v {
                            a[(j_sdr * stride as i32 + d_sd) as usize]
                        } else {
                            mx.dp[y][(j_sdr * stride as i32 + d_sd) as usize]
                        };
                        a[idx] = flogsum(a[idx], child + tsc_v[yo]);
                    }
                    a[idx] += esc_v[dsq[j as usize] as usize];
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else if stt != B_ST {
            for j in sdr..=l {
                let j_sdr = j - sdr;
                for d in sd..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        a[idx] = flogsum(a[idx], mx.dp[y][(j_sdr * stride as i32 + (d - sd)) as usize] + tsc_v[yo]);
                    }
                }
            }
            match stt {
                x if x == ML_ST => {
                    for j in 0..=l {
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            a[idx] += esc_v[dsq[(j - d + 1) as usize] as usize];
                        }
                    }
                }
                x if x == MR_ST => {
                    for j in 0..=l {
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            a[idx] += esc_v[dsq[j as usize] as usize];
                        }
                    }
                }
                x if x == MP_ST => {
                    for j in 0..=l {
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            let li = dsq[(j - d + 1) as usize] as usize;
                            let ri = dsq[j as usize] as usize;
                            a[idx] += esc_v[li * kp + ri];
                        }
                    }
                }
                _ => {}
            }
            for j in 0..=ll {
                for d in 0..=j {
                    let idx = j * stride + d;
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else {
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            for j in 0..=l {
                for d in 0..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    for k in 0..=d {
                        a[idx] = flogsum(
                            a[idx],
                            mx.dp[y][((j - k) * stride as i32 + (d - k)) as usize]
                                + mx.dp[z][(j * stride as i32 + k) as usize],
                        );
                    }
                }
            }
        }

        let root_idx = (l * stride as i32 + l) as usize;
        if local_begin && not_impossible(cm.beginsc[v]) {
            bsc = flogsum(bsc, a[root_idx] + cm.beginsc[v]);
        }
        mx.dp[v] = a;
    }

    let root_idx = (l * stride as i32 + l) as usize;
    mx.dp[0][root_idx] = flogsum(mx.dp[0][root_idx], bsc);
    let sc = mx.dp[0][root_idx];
    (mx, sc)
}

// ============================================================================
// cm_OutsideAlign (cm_dpalign.c:4013)
// ============================================================================
fn cm_outside_align(cm: &CM, dsq: &[u8], l: i32, ins_mx: &CmMx) -> CmMx {
    let m = cm.m as usize;
    let ll = l as usize;
    let mut mx = CmMx::new(m, ll);
    let stride = mx.stride;
    let kp = ALPHABET_SIZE_P;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let alpha = &ins_mx.dp;

    // beta[0][L][L] = 0 (cm_dpalign.c:4046)
    mx.dp[0][(l * stride as i32 + l) as usize] = 0.0;
    // local begin cells (cm_dpalign.c:4049)
    if local_begin {
        for v in 1..m {
            mx.dp[v][(l * stride as i32 + l) as usize] = cm.beginsc[v];
        }
    }

    for v in 1..m {
        let stt = cm.sttype[v] as i32;
        let stid = cm.stid[v] as i32;

        if stid == BEGL_S {
            let y = cm.plast[v] as usize; // parent bifurcation
            let z = cm.cnum[y] as usize; // right S
            for j in 0..=l {
                for d in 0..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    let mut acc = mx.dp[v][idx];
                    for k in 0..=(l - j) {
                        acc = flogsum(
                            acc,
                            mx.dp[y][((j + k) * stride as i32 + (d + k)) as usize]
                                + alpha[z][((j + k) * stride as i32 + k) as usize],
                        );
                    }
                    mx.dp[v][idx] = acc;
                }
            }
        } else if stid == BEGR_S {
            let y = cm.plast[v] as usize; // parent bifurcation
            let z = cm.cfirst[y] as usize; // left S
            for j in 0..=l {
                for d in 0..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    let mut acc = mx.dp[v][idx];
                    for k in 0..=(j - d) {
                        acc = flogsum(
                            acc,
                            mx.dp[y][(j * stride as i32 + (d + k)) as usize]
                                + alpha[z][((j - d) * stride as i32 + k) as usize],
                        );
                    }
                    mx.dp[v][idx] = acc;
                }
            }
        } else {
            for j in (0..=l).rev() {
                let mut i = 1;
                let mut d = j;
                while d >= 0 {
                    let idx = (j * stride as i32 + d) as usize;
                    let mut acc = mx.dp[v][idx];
                    // parents y (cm_dpalign.c:4085)
                    let plast = cm.plast[v];
                    let pnum = cm.pnum[v];
                    let mut y = plast;
                    while y > plast - pnum {
                        let yu = y as usize;
                        let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                        let yst = cm.sttype[yu] as i32;
                        let sd = state_delta(yst);
                        let sdr = state_right_delta(yst);
                        let byidx = ((j + sdr) * stride as i32 + (d + sd)) as usize;
                        match yst {
                            x if x == MP_ST => {
                                if !(j == l || d == j) {
                                    let escore = cm.oesc[yu][dsq[(i - 1) as usize] as usize * kp
                                        + dsq[(j + 1) as usize] as usize];
                                    acc = flogsum(acc, mx.dp[yu][byidx] + cm.tsc[yu][voffset] + escore);
                                }
                            }
                            x if x == ML_ST || x == IL_ST => {
                                if d != j {
                                    let escore = cm.oesc[yu][dsq[(i - 1) as usize] as usize];
                                    acc = flogsum(acc, mx.dp[yu][byidx] + cm.tsc[yu][voffset] + escore);
                                }
                            }
                            x if x == MR_ST || x == IR_ST => {
                                if j != l {
                                    let escore = cm.oesc[yu][dsq[(j + 1) as usize] as usize];
                                    acc = flogsum(acc, mx.dp[yu][byidx] + cm.tsc[yu][voffset] + escore);
                                }
                            }
                            x if x == S_ST || x == E_ST || x == D_ST => {
                                acc = flogsum(acc, mx.dp[yu][byidx] + cm.tsc[yu][voffset]);
                            }
                            _ => {}
                        }
                        y -= 1;
                    }
                    if acc < IMPOSSIBLE {
                        acc = IMPOSSIBLE;
                    }
                    mx.dp[v][idx] = acc;
                    i += 1;
                    d -= 1;
                }
            }
        }

        // v -> EL transitions (cm_dpalign.c:4124)
        if local_end && not_impossible(cm.endsc[v]) {
            let sdr = state_right_delta(stt);
            let sd = state_delta(stt);
            for j in 0..=l {
                for d in 0..=j {
                    let i = j - d + 1;
                    let elidx = (j * stride as i32 + d) as usize;
                    let byidx = ((j + sdr) * stride as i32 + (d + sd)) as usize;
                    match stt {
                        x if x == MP_ST => {
                            if !(j == l || d == j) {
                                let escore = cm.oesc[v][dsq[(i - 1) as usize] as usize * kp
                                    + dsq[(j + 1) as usize] as usize];
                                mx.dp[m][elidx] = flogsum(mx.dp[m][elidx], mx.dp[v][byidx] + cm.endsc[v] + escore);
                            }
                        }
                        x if x == ML_ST || x == IL_ST => {
                            if d != j {
                                let escore = cm.oesc[v][dsq[(i - 1) as usize] as usize];
                                mx.dp[m][elidx] = flogsum(mx.dp[m][elidx], mx.dp[v][byidx] + cm.endsc[v] + escore);
                            }
                        }
                        x if x == MR_ST || x == IR_ST => {
                            if j != l {
                                let escore = cm.oesc[v][dsq[(j + 1) as usize] as usize];
                                mx.dp[m][elidx] = flogsum(mx.dp[m][elidx], mx.dp[v][byidx] + cm.endsc[v] + escore);
                            }
                        }
                        x if x == S_ST || x == D_ST || x == B_ST || x == E_ST => {
                            mx.dp[m][elidx] = flogsum(mx.dp[m][elidx], mx.dp[v][byidx] + cm.endsc[v]);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // EL->EL left-emitting (cm_dpalign.c:4166)
    if local_end {
        for j in (1..=l).rev() {
            for d in (0..=(j - 1)).rev() {
                let idx = (j * stride as i32 + d) as usize;
                let idx1 = (j * stride as i32 + (d + 1)) as usize;
                mx.dp[m][idx] = flogsum(mx.dp[m][idx], mx.dp[m][idx1] + cm.el_selfsc);
            }
        }
    }
    mx
}

// ============================================================================
// cm_Posterior (cm_dpalign.c:4819)
// ============================================================================
fn cm_posterior(cm: &CM, l: i32, ins_mx: &CmMx, out_mx: &CmMx) -> CmMx {
    let m = cm.m as usize;
    let ll = l as usize;
    let mut post = CmMx::new(m, ll);
    let stride = post.stride;
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let sc = ins_mx.dp[0][(l * stride as i32 + l) as usize];
    let vmax = if local_end { m } else { m - 1 };
    for v in (0..=vmax).rev() {
        for j in 0..=ll {
            for d in 0..=j {
                let idx = j * stride + d;
                post.dp[v][idx] = ins_mx.dp[v][idx] + out_mx.dp[v][idx] - sc;
            }
        }
    }
    post
}

// ============================================================================
// cm_EmitterPosterior (cm_dpalign.c:5044)
// ============================================================================
fn cm_emitter_posterior(cm: &CM, l: i32, post: &CmMx) -> CmEmitMx {
    let m = cm.m as usize;
    let ll = l as usize;
    let stride = post.stride;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    // allocate l_pp / r_pp only for emitting states (else None)
    let mut l_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut r_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            l_pp[v] = Some(vec![IMPOSSIBLE; ll + 1]);
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            r_pp[v] = Some(vec![IMPOSSIBLE; ll + 1]);
        }
    }
    if local_end {
        l_pp[m] = Some(vec![IMPOSSIBLE; ll + 1]);
    }

    // Step 1 (cm_dpalign.c:5062)
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            let lp = l_pp[v].as_mut().unwrap();
            for j in 1..=l {
                let mut i = j - sd + 1;
                for d in sd..=j {
                    lp[i as usize] = flogsum(lp[i as usize], post.dp[v][(j * stride as i32 + d) as usize]);
                    i -= 1;
                }
            }
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            let rp = r_pp[v].as_mut().unwrap();
            for j in 1..=l {
                for d in sd..=j {
                    rp[j as usize] = flogsum(rp[j as usize], post.dp[v][(j * stride as i32 + d) as usize]);
                }
            }
        }
    }
    // EL contribution (cm_dpalign.c:5082)
    if local_end {
        let lp = l_pp[m].as_mut().unwrap();
        for j in 1..=l {
            let mut i = j;
            for d in 1..=j {
                lp[i as usize] = flogsum(lp[i as usize], post.dp[m][(j * stride as i32 + d) as usize]);
                i -= 1;
            }
        }
    }

    // Step 2: normalize (cm_dpalign.c:5100)
    let mut sum = vec![IMPOSSIBLE; ll + 1];
    for v in 0..=m {
        if let Some(lp) = &l_pp[v] {
            for i in 1..=ll {
                sum[i] = flogsum(sum[i], lp[i]);
            }
        }
        if let Some(rp) = &r_pp[v] {
            for j in 1..=ll {
                sum[j] = flogsum(sum[j], rp[j]);
            }
        }
    }
    for v in 0..=m {
        if let Some(lp) = l_pp[v].as_mut() {
            for i in 1..=ll {
                lp[i] -= sum[i];
            }
        }
        if let Some(rp) = r_pp[v].as_mut() {
            for j in 1..=ll {
                rp[j] -= sum[j];
            }
        }
    }

    // Step 3: combine MATP_MP (v) with MATP_ML (v+1) and MATP_MR (v+2)
    // (cm_dpalign.c:5146)
    for v in 0..=m {
        if v < m && cm.sttype[v] as i32 == MP_ST {
            for i in 1..=ll {
                let a = l_pp[v].as_ref().unwrap()[i];
                let bb = l_pp[v + 1].as_ref().unwrap()[i];
                let s = flogsum(a, bb);
                l_pp[v].as_mut().unwrap()[i] = s;
                l_pp[v + 1].as_mut().unwrap()[i] = s;
            }
            for j in 1..=ll {
                let a = r_pp[v].as_ref().unwrap()[j];
                let bb = r_pp[v + 2].as_ref().unwrap()[j];
                let s = flogsum(a, bb);
                r_pp[v].as_mut().unwrap()[j] = s;
                r_pp[v + 2].as_mut().unwrap()[j] = s;
            }
        }
    }

    CmEmitMx { l_pp, r_pp, sum }
}

// ============================================================================
// cm_InitializeOptAccShadowDZero (cm_dpalign.c:5646)
// ============================================================================
pub(crate) fn cm_init_optacc_shadow_dzero(cm: &CM, yshadow: &mut [Vec<i32>], l: i32, stride: usize) {
    let m = cm.m as usize;
    let have_el = cm.flags & CMH_LOCAL_END != 0;
    let mut esc: Vec<f32> = vec![0.0; m];
    let endsc: f32;
    if have_el {
        let mut v0 = 0usize;
        while !not_impossible(cm.endsc[v0]) {
            v0 += 1;
        }
        endsc = cm.endsc[v0];
    } else {
        endsc = IMPOSSIBLE;
    }

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        if stt == E_ST {
            if have_el {
                esc[v] = 0.0;
            }
        } else if stt == B_ST {
            if have_el {
                let y = cm.cfirst[v] as usize;
                let z = cm.cnum[v] as usize;
                esc[v] = esc[y] + esc[z];
            }
        } else {
            // one and only child y with StateDelta(y)==0
            let mut y = cm.cfirst[v] as usize;
            while state_delta(cm.sttype[y] as i32) != 0 {
                y += 1;
            }
            let mut yoffset = (y as i32) - cm.cfirst[v];
            if have_el {
                esc[v] = esc[y] + cm.tsc[v][yoffset as usize];
                if endsc > esc[v] {
                    yoffset = USED_EL;
                }
            }
            for j in sd..=l {
                yshadow[v][(j * stride as i32 + sd) as usize] = yoffset;
            }
        }
    }
}

// ============================================================================
// cm_OptAccAlign (cm_dpalign.c:2205)
// ============================================================================
/// Returns (mx, shadow, b, pp).
fn cm_optacc_align(cm: &CM, l: i32, emit_mx: &CmEmitMx) -> (CmMx, CmShadowMx, i32, f32) {
    let m = cm.m as usize;
    let ll = l as usize;
    let mut mx = CmMx::new(m, ll);
    let stride = mx.stride;
    let ncells = stride * stride;
    let mut yshadow: Vec<Vec<i32>> = vec![vec![USED_EL; ncells]; m];
    let mut kshadow: Vec<Vec<i32>> = vec![vec![0i32; ncells]; m];

    let mut b: i32 = -1;
    let mut bsc = IMPOSSIBLE;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let have_el = cm.flags & CMH_LOCAL_END != 0;

    // initialize yshadow for d==0 (cm_dpalign.c:2247)
    cm_init_optacc_shadow_dzero(cm, &mut yshadow, l, stride);

    // EL state (cm_dpalign.c:2250)
    if have_el {
        if let Some(lpm) = &emit_mx.l_pp[m] {
            for j in 0..=l {
                mx.dp[m][(j * stride as i32) as usize] = lpm[0];
                let mut i = j;
                for d in 1..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    let idxm1 = (j * stride as i32 + (d - 1)) as usize;
                    mx.dp[m][idx] = flogsum(mx.dp[m][idxm1], lpm[i as usize]);
                    i -= 1;
                }
            }
        }
    }

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let endsc_v = cm.endsc[v];

        let mut a = vec![IMPOSSIBLE; ncells];
        let mut ys = yshadow[v].clone(); // preserve d==0 init
        let mut ks = vec![0i32; ncells];

        // re-init if local end from v (cm_dpalign.c:2267): copy from EL deck
        if have_el && not_impossible(endsc_v) {
            for j in 0..=l {
                for d in sd..=j {
                    a[(j * stride as i32 + d) as usize] =
                        mx.dp[m][((j - sdr) * stride as i32 + (d - sd)) as usize];
                }
            }
        }

        if stt == IL_ST {
            let lp = emit_mx.l_pp[v].as_ref().unwrap();
            for j in 1..=l {
                for d in 1..=j {
                    let d_sd = d - sd;
                    let i = j - d + 1; // == i in C (i starts j, decrements): l_pp[v][i], i=j-d+1
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        let sc = if y == v {
                            a[(j * stride as i32 + d_sd) as usize]
                        } else {
                            mx.dp[y][(j * stride as i32 + d_sd) as usize]
                        };
                        if sc > a[idx] {
                            a[idx] = sc;
                            ys[idx] = yo as i32;
                        }
                    }
                    a[idx] = flogsum(a[idx], lp[i as usize]);
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                    if !have_el && ys[idx] == USED_EL && d > sd {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else if stt == IR_ST {
            let rp = emit_mx.r_pp[v].as_ref().unwrap();
            for j in 1..=l {
                let j_sdr = j - sdr;
                for d in 1..=j {
                    let d_sd = d - sd;
                    let idx = (j * stride as i32 + d) as usize;
                    for yo in 0..cnum {
                        let y = cfirst + yo;
                        let sc = if y == v {
                            a[(j_sdr * stride as i32 + d_sd) as usize]
                        } else {
                            mx.dp[y][(j_sdr * stride as i32 + d_sd) as usize]
                        };
                        if sc > a[idx] {
                            a[idx] = sc;
                            ys[idx] = yo as i32;
                        }
                    }
                    a[idx] = flogsum(a[idx], rp[j as usize]);
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                    if !have_el && ys[idx] == USED_EL && d > sd {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else if stt != B_ST {
            // ML, MP, MR, D, S (cm_dpalign.c:2330)
            for yo in 0..cnum {
                let y = cfirst + yo;
                for j in sdr..=l {
                    let j_sdr = j - sdr;
                    for d in sd..=j {
                        let idx = (j * stride as i32 + d) as usize;
                        let sc = mx.dp[y][(j_sdr * stride as i32 + (d - sd)) as usize];
                        if sc > a[idx] {
                            a[idx] = sc;
                            ys[idx] = yo as i32;
                        }
                    }
                }
            }
            match stt {
                x if x == ML_ST => {
                    let lp = emit_mx.l_pp[v].as_ref().unwrap();
                    for j in 1..=l {
                        let mut i = j;
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            a[idx] = flogsum(a[idx], lp[i as usize]);
                            i -= 1;
                        }
                    }
                }
                x if x == MR_ST => {
                    let rp = emit_mx.r_pp[v].as_ref().unwrap();
                    for j in 1..=l {
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            a[idx] = flogsum(a[idx], rp[j as usize]);
                        }
                    }
                }
                x if x == MP_ST => {
                    let lp = emit_mx.l_pp[v].as_ref().unwrap();
                    let rp = emit_mx.r_pp[v].as_ref().unwrap();
                    for j in 2..=l {
                        let mut i = j - 1;
                        for d in sd..=j {
                            let idx = (j * stride as i32 + d) as usize;
                            a[idx] = flogsum(a[idx], flogsum(lp[i as usize], rp[j as usize]));
                            i -= 1;
                        }
                    }
                }
                _ => {}
            }
            for j in 0..=ll {
                for d in 0..=j {
                    let idx = j * stride + d;
                    if a[idx] < IMPOSSIBLE {
                        a[idx] = IMPOSSIBLE;
                    }
                }
            }
            if !have_el && sd > 0 {
                for j in 0..=l {
                    for d in (sd + 1)..=j {
                        let idx = (j * stride as i32 + d) as usize;
                        if ys[idx] == USED_EL {
                            a[idx] = IMPOSSIBLE;
                        }
                    }
                }
            }
        } else {
            // B_st (cm_dpalign.c:2394)
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            for j in 0..=l {
                for d in 0..=j {
                    let idx = (j * stride as i32 + d) as usize;
                    for k in 0..=d {
                        let lft = mx.dp[y][((j - k) * stride as i32 + (d - k)) as usize];
                        let rgt = mx.dp[z][(j * stride as i32 + k) as usize];
                        let sc = flogsum(lft, rgt);
                        if sc > a[idx]
                            && ((d == k) || not_impossible(lft))
                            && ((k == 0) || not_impossible(rgt))
                        {
                            a[idx] = sc;
                            ks[idx] = k;
                        }
                    }
                }
            }
        }

        // local begins (cm_dpalign.c:2439)
        let root_idx = (l * stride as i32 + l) as usize;
        if local_begin && not_impossible(cm.beginsc[v]) && a[root_idx] > bsc {
            b = v as i32;
            bsc = a[root_idx];
        }

        mx.dp[v] = a;
        yshadow[v] = ys;
        kshadow[v] = ks;
    }

    // fold local begin (cm_dpalign.c:2458)
    let root_idx = (l * stride as i32 + l) as usize;
    if not_impossible(bsc) && local_begin {
        mx.dp[0][root_idx] = bsc;
        yshadow[0][root_idx] = USED_LOCAL_BEGIN;
    }
    let sc = mx.dp[0][root_idx];
    let pp = (sre_exp2(sc) / l as f64) as f32;
    let sh = CmShadowMx { yshadow, kshadow, stride };
    (mx, sh, b, pp)
}

// ============================================================================
// cm_alignT (cm_dpalign.c:117) — traceback into a Parsetree.
// ============================================================================
fn cm_align_t(cm: &CM, l: i32, sh: &CmShadowMx, b: i32) -> Parsetree {
    let stride = sh.stride;
    let mut tr = Parsetree::new(100);
    // init: attach root S (cm_dpalign.c:136)
    tr.add_node(1, l, 0, -1, -1, -1);
    // pda: (j, k, bifparent_tridx)
    let mut pda: Vec<(i32, i32, i32)> = Vec::new();
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = l;
    let mut d: i32 = l;

    loop {
        // E or EL (v==M) : swing over to pending right subtree, or finish.
        if v == cm.m || cm.sttype[v as usize] as i32 == E_ST || cm.sttype[v as usize] as i32 == EL_ST {
            match pda.pop() {
                None => break,
                Some((sj, sd_saved, bifparent)) => {
                    d = sd_saved;
                    j = sj;
                    let bstate = tr.state[bifparent as usize];
                    let y = cm.cnum[bstate as usize];
                    i = j - d + 1;
                    let idx = tr.add_node(i, j, y, -1, -1, bifparent);
                    tr.nxtr[bifparent as usize] = idx;
                    v = y;
                    continue;
                }
            }
        }
        let vu = v as usize;
        let stt = cm.sttype[vu] as i32;
        if stt == B_ST {
            let k = sh.kshadow[vu][(j * stride as i32 + d) as usize];
            let parent = tr.n - 1;
            pda.push((j, k, parent));
            j -= k;
            d -= k;
            i = j - d + 1;
            let y = cm.cfirst[vu];
            let idx = tr.add_node(i, j, y, -1, -1, parent);
            tr.nxtl[parent as usize] = idx;
            v = y;
        } else {
            let yoffset = sh.yshadow[vu][(j * stride as i32 + d) as usize];
            match stt {
                x if x == MP_ST => { i += 1; j -= 1; }
                x if x == ML_ST => { i += 1; }
                x if x == MR_ST => { j -= 1; }
                x if x == IL_ST => { i += 1; }
                x if x == IR_ST => { j -= 1; }
                _ => {} // D, S
            }
            d = j - i + 1;
            let parent = tr.n - 1;
            let ny = if yoffset == USED_EL {
                cm.m
            } else if yoffset == USED_LOCAL_BEGIN {
                b
            } else {
                cm.cfirst[vu] + yoffset
            };
            let idx = tr.add_node(i, j, ny, -1, -1, parent);
            tr.nxtl[parent as usize] = idx;
            v = ny;
        }
    }
    tr
}

// ============================================================================
// cm_PostCode (cm_dpalign.c:5450). Returns (ppstr[0..L-1], avgpp).
// ============================================================================
fn fscore2postcode(sc: f32) -> u8 {
    // FScore2Prob(sc, 1.) (cm_dpalign.c:5441)
    let p: f32 = if !not_impossible(sc) { 0.0 } else { sre_exp2(sc) as f32 };
    let pd = p as f64;
    if pd + 0.05 >= 1.0 {
        b'*'
    } else {
        (((pd + 0.05) * 10.0) as i32 as u8).wrapping_add(b'0')
    }
}
fn cm_postcode(cm: &CM, l: i32, emit_mx: &CmEmitMx, tr: &Parsetree) -> (Vec<u8>, f32) {
    let mut ppstr = vec![0u8; l as usize];
    let mut sum_logp = IMPOSSIBLE;
    for x in 0..tr.n as usize {
        let v = tr.state[x] as usize;
        let i = tr.emitl[x];
        let j = tr.emitr[x];
        let stt = if v == cm.m as usize { EL_ST } else { cm.sttype[v] as i32 };
        if stt == EL_ST {
            for r in i..=j {
                let val = emit_mx.l_pp[v].as_ref().unwrap()[r as usize];
                ppstr[(r - 1) as usize] = fscore2postcode(val);
                sum_logp = flogsum(sum_logp, val);
            }
        }
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            let val = emit_mx.l_pp[v].as_ref().unwrap()[i as usize];
            ppstr[(i - 1) as usize] = fscore2postcode(val);
            sum_logp = flogsum(sum_logp, val);
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            let val = emit_mx.r_pp[v].as_ref().unwrap()[j as usize];
            ppstr[(j - 1) as usize] = fscore2postcode(val);
            sum_logp = flogsum(sum_logp, val);
        }
    }
    let avgp = (sre_exp2(sum_logp) / l as f64) as f32;
    (ppstr, avgp)
}

// ============================================================================
// cm_Align (cm_dpalign.c:644) driver.
// ============================================================================
/// Align a full sequence 1..L. `do_optacc` selects optimal-accuracy (default),
/// else CYK. Returns (parsetree, optional PP string [0..L-1], score/avgpp).
pub fn cm_align(cm: &CM, dsq: &[u8], l: i32, do_optacc: bool, want_pp: bool) -> (Parsetree, Option<Vec<u8>>, f32) {
    let do_post = do_optacc || want_pp;
    let mut emit_mx: Option<CmEmitMx> = None;
    let mut ins_sc = 0.0f32;
    if do_post {
        let (ins, sc) = cm_inside_align(cm, dsq, l);
        ins_sc = sc;
        let out = cm_outside_align(cm, dsq, l, &ins);
        let post = cm_posterior(cm, l, &ins, &out);
        emit_mx = Some(cm_emitter_posterior(cm, l, &post));
    }

    let (sh, b, cyk_sc);
    if do_optacc {
        let (_mx, shx, bx, _pp) = cm_optacc_align(cm, l, emit_mx.as_ref().unwrap());
        sh = shx;
        b = bx;
        cyk_sc = ins_sc;
    } else {
        let (_mx, shx, bx, scx) = cm_cyk_inside_align(cm, dsq, l);
        sh = shx;
        b = bx;
        cyk_sc = scx;
    }
    let tr = cm_align_t(cm, l, &sh, b);

    let ppstr = if want_pp {
        let (pp, _avg) = cm_postcode(cm, l, emit_mx.as_ref().unwrap(), &tr);
        Some(pp)
    } else {
        None
    };
    (tr, ppstr, cyk_sc)
}

// ============================================================================
// Stochastic parsetree sampling (`cmalign --sample`). Faithful port of:
//   cm_StochasticParsetree (cm_parsetree.c:2789)
//   sample_helper          (cm_parsetree.c:4711)
//   get_femission_score    (cm_parsetree.c:4646)
// plus the esl_vectorops.c / esl_random.c primitives they call. The sampled
// parsetree is drawn from a pre-filled non-banded float Inside matrix; the
// exact esl_random draw order must match C for byte-parity.
// ============================================================================

// esl_vectorops.c:esl_vec_FMax
fn esl_vec_fmax(v: &[f32]) -> f32 {
    let mut best = v[0];
    for &x in &v[1..] {
        if x > best {
            best = x;
        }
    }
    best
}

// esl_vectorops.c:esl_vec_FSum (Kahan compensated summation)
fn esl_vec_fsum(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for &vi in v {
        let y = vi - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

// esl_vectorops.c:esl_vec_FLogSum
fn esl_vec_flogsum(v: &[f32]) -> f32 {
    let max = esl_vec_fmax(v);
    if max == f32::INFINITY {
        return f32::INFINITY;
    }
    let mut sum = 0.0f32;
    for &vi in v {
        // C: `if (vec[i] > max - 50.)` — `50.` is a double literal, so the
        // comparison is evaluated in double precision (vec[i] and max promote).
        if (vi as f64) > (max as f64) - 50.0 {
            sum += (vi - max).exp(); // expf, float arg
        }
    }
    sum.ln() + max // logf
}

// esl_vectorops.c:esl_vec_FLogNorm = FLogSum; FIncrement(-denom); FExp; FNorm
fn esl_vec_flognorm(v: &mut [f32]) {
    let denom = esl_vec_flogsum(v);
    for x in v.iter_mut() {
        *x += -1.0 * denom;
    }
    for x in v.iter_mut() {
        *x = x.exp(); // FExp (expf)
    }
    let sum = esl_vec_fsum(v);
    if sum != 0.0 {
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

// esl_random.c:esl_rnd_FChoose — accumulate in double precision; one esl_random
// draw per call (roll is a [0,1) double).
fn esl_rnd_fchoose(r: &mut EslRandom, p: &[f32]) -> usize {
    let roll = r.random(); // f64 in [0,1)
    let mut norm = 0.0f64;
    for &pi in p {
        norm += pi as f64;
    }
    let mut sum = 0.0f64;
    for (i, &pi) in p.iter().enumerate() {
        sum += pi as f64;
        if roll < sum / norm {
            return i;
        }
    }
    // C esl_fatal("unreached code...") here; never reached in practice.
    p.len() - 1
}

// cm_parsetree.c:sample_helper — normalize a non-normalized log_2 prob vector,
// then draw a valid index. Returns None iff all elements are IMPOSSIBLE.
// Exposed pub(crate) so the truncated HB stochastic traceback (cm_trunc.rs) can
// reuse the exact same RNG-consuming normalization + FChoose.
pub(crate) fn sample_helper(r: &mut EslRandom, pa: &mut [f32]) -> Option<usize> {
    let n = pa.len();
    let mut valid = vec![false; n];
    for i in 0..n {
        if not_impossible(pa[i]) {
            valid[i] = true;
        }
    }
    if !valid.iter().any(|&b| b) {
        return None;
    }
    let maxsc = esl_vec_fmax(pa);
    for x in pa.iter_mut() {
        *x += -1.0 * maxsc; // FIncrement(pA, n, -maxsc)
    }
    let ln2 = 2.0f64.ln() as f32; // (float) log(2.)
    for x in pa.iter_mut() {
        *x *= ln2; // FScale(pA, n, log(2.))
    }
    esl_vec_flognorm(pa);
    loop {
        let i = esl_rnd_fchoose(r, pa);
        if valid[i] {
            return Some(i);
        }
    }
}

// cm_parsetree.c:get_femission_score
fn get_femission_score(cm: &CM, dsq: &[u8], v: usize, i: i32, j: i32) -> f32 {
    let stt = cm.sttype[v] as i32;
    if stt == ML_ST || stt == IL_ST {
        cm.oesc[v][dsq[i as usize] as usize]
    } else if stt == MR_ST || stt == IR_ST {
        cm.oesc[v][dsq[j as usize] as usize]
    } else if stt == MP_ST {
        cm.oesc[v][dsq[i as usize] as usize * ALPHABET_SIZE_P + dsq[j as usize] as usize]
    } else {
        0.0
    }
}

// cm_parsetree.c:cm_StochasticParsetree — sample a parsetree from Inside matrix
// `mx` (filled by cm_inside_align). Returns (parsetree, sampled-parse score fsc).
fn cm_stochastic_parsetree(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    mx: &CmMx,
    r: &mut EslRandom,
) -> (Parsetree, f32) {
    let stride = mx.stride as i32;
    let m = cm.m; // i32 (EL deck index)
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let alpha = &mx.dp;

    // Create the parse tree and attach the root S (C: InsertTraceNode 1..L, 0).
    let mut tr = Parsetree::new(100);
    tr.add_node(1, l, 0, -1, -1, -1);

    // pda stack of ints, pushed j, k, parent-tridx (popped in reverse order).
    let mut pda: Vec<i32> = Vec::new();

    let mut v: i32 = 0;
    let mut j: i32 = l;
    let mut d: i32 = l;
    let mut i: i32 = 1;
    let mut fsc: f32 = 0.0;

    loop {
        let is_el = v == m;
        let stt = if is_el { EL_ST } else { cm.sttype[v as usize] as i32 };

        if !is_el && stt == B_ST {
            let y = cm.cfirst[v as usize]; // left child (BEGL_S)
            let z = cm.cnum[v as usize]; // right child (BEGR_S)
            let cur = (d + 1) as usize;
            let mut pa = vec![IMPOSSIBLE; cur];
            for k in 0..=d {
                pa[k as usize] = alpha[y as usize][((j - k) * stride + (d - k)) as usize]
                    + alpha[z as usize][(j * stride + k) as usize];
            }
            let choice =
                sample_helper(r, &mut pa).expect("cm_StochasticParsetree: no valid B_st k");
            let k = choice as i32;
            // remember (end j, subseq length k, trace index of parent B)
            pda.push(j);
            pda.push(k);
            pda.push(tr.n - 1);
            j -= k;
            d -= k;
            i = j - d + 1;
            let parent = tr.n - 1;
            let idx = tr.add_node(i, j, y, -1, -1, parent);
            tr.nxtl[parent as usize] = idx;
            v = y;
        } else if is_el || stt == E_ST {
            // E or EL: swing over to a pending right subtree, or finish.
            if pda.is_empty() {
                break;
            }
            let bifparent = pda.pop().unwrap();
            d = pda.pop().unwrap();
            j = pda.pop().unwrap();
            v = tr.state[bifparent as usize]; // recover the B state
            let y = cm.cnum[v as usize]; // right S
            i = j - d + 1;
            let idx = tr.add_node(i, j, y, -1, -1, bifparent);
            tr.nxtr[bifparent as usize] = idx;
            v = y;
        } else {
            let mut yoffset: i32;
            let mut b: i32 = -1;
            if v > 0 || !local_begin {
                fsc += get_femission_score(cm, dsq, v as usize, i, j);
                let sd = state_delta(stt);
                let sdr = state_right_delta(stt);
                let cnum = cm.cnum[v as usize];
                let mut el_is_possible = false;
                let mut cur = cnum as usize;
                if local_end && not_impossible(cm.endsc[v as usize]) {
                    el_is_possible = true;
                    cur += 1;
                }
                let mut pa = vec![IMPOSSIBLE; cur];
                for yo in 0..cnum {
                    let y = cm.cfirst[v as usize] + yo;
                    pa[yo as usize] = cm.tsc[v as usize][yo as usize]
                        + alpha[y as usize][((j - sdr) * stride + (d - sd)) as usize];
                }
                if el_is_possible {
                    pa[cur - 1] =
                        cm.endsc[v as usize] + alpha[m as usize][(j * stride + d) as usize];
                }
                let choice = sample_helper(r, &mut pa)
                    .expect("cm_StochasticParsetree: no valid non-B_st transition");
                yoffset = choice as i32;
                if yoffset < cnum {
                    fsc += cm.tsc[v as usize][yoffset as usize];
                } else {
                    yoffset = USED_EL; // chose EL
                    fsc += cm.endsc[v as usize] + cm.el_selfsc * (d - sd) as f32;
                }
            } else {
                // v == 0 && local begins are on: sample the local begin state.
                let cur = m as usize;
                let mut pa = vec![IMPOSSIBLE; cur];
                for y in 0..m {
                    if not_impossible(cm.beginsc[y as usize]) {
                        pa[y as usize] =
                            cm.beginsc[y as usize] + alpha[y as usize][(j * stride + d) as usize];
                    }
                }
                let choice = sample_helper(r, &mut pa)
                    .expect("cm_StochasticParsetree: no valid local begin");
                b = choice as i32;
                fsc += cm.beginsc[b as usize];
                yoffset = USED_LOCAL_BEGIN;
            }

            // adjust i and j based on state type (C switch)
            match stt {
                x if x == D_ST => {}
                x if x == MP_ST => {
                    i += 1;
                    j -= 1;
                }
                x if x == ML_ST => i += 1,
                x if x == MR_ST => j -= 1,
                x if x == IL_ST => i += 1,
                x if x == IR_ST => j -= 1,
                x if x == S_ST => {}
                _ => panic!("cm_StochasticParsetree: inconceivable state type {}", stt),
            }
            d = j - i + 1;
            let parent = tr.n - 1;
            if yoffset == USED_EL {
                let idx = tr.add_node(i, j, m, -1, -1, parent);
                tr.nxtl[parent as usize] = idx;
                v = m;
            } else if yoffset == USED_LOCAL_BEGIN {
                let idx = tr.add_node(i, j, b, -1, -1, parent);
                tr.nxtl[parent as usize] = idx;
                v = b;
            } else {
                let y = cm.cfirst[v as usize] + yoffset;
                let idx = tr.add_node(i, j, y, -1, -1, parent);
                tr.nxtl[parent as usize] = idx;
                v = y;
            }
        }
    }
    (tr, fsc)
}

// cm_Align (cm_dpalign.c:644) driver, do_sample==TRUE branch. Fills Inside,
// samples a parsetree from it, then (if want_pp) fills Outside/Posterior/Emit
// and computes the PP string via cm_PostCode against the sampled parse.
pub fn cm_align_sample(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    want_pp: bool,
    r: &mut EslRandom,
) -> (Parsetree, Option<Vec<u8>>, f32) {
    let (ins, _ins_sc) = cm_inside_align(cm, dsq, l);
    let (tr, fsc) = cm_stochastic_parsetree(cm, dsq, l, &ins, r);
    let ppstr = if want_pp {
        let out = cm_outside_align(cm, dsq, l, &ins);
        let post = cm_posterior(cm, l, &ins, &out);
        let emit_mx = cm_emitter_posterior(cm, l, &post);
        let (pp, _avg) = cm_postcode(cm, l, &emit_mx, &tr);
        Some(pp)
    } else {
        None
    };
    (tr, ppstr, fsc)
}

// cm_AlignHB (cm_dpalign.c:740) do_sample branch: fill HB Inside, sample a
// parsetree via cm_StochasticParsetreeHB, then (if want_pp, i.e. have_ppstr)
// run the do_post chain Outside/Posterior/EmitterPosterior/PostCodeHB against
// the sampled parse to produce the #=GR PP string + avg PP (C cm_AlignHB
// do_post). Used by non-truncated HB `cmalign --sample --notrunc` and by
// cmbuild --refine --gibbs --notrunc. Returns (parsetree, ppstr, sampled score, avgpp).
pub(crate) fn cm_align_sample_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[u8],
    l: i32,
    want_pp: bool,
    r: &mut EslRandom,
) -> (Parsetree, Option<Vec<u8>>, f32, f32) {
    // C: if(do_post || do_sample) fill Inside; if(do_sample) sample from it.
    let (ins, _ins_sc) = cm_inside_align_hb(cm, cp9b, dsq, l);
    let (tr, fsc) = cm_stochastic_parsetree_hb(cm, cp9b, dsq, l, &ins, r);
    let mut avgpp = 0.0f32;
    let ppstr = if want_pp {
        // C do_post: Outside then Posterior then EmitterPosterior, then PostCodeHB.
        let out = cm_outside_align_hb(cm, cp9b, dsq, l, &ins);
        let post = cm_posterior_hb(cm, cp9b, l, &ins, &out);
        let emit_mx = cm_emitter_posterior_hb(cm, cp9b, l, &post);
        let (pp, avg) = cm_postcode_hb(cm, cp9b, l, &emit_mx, &tr);
        avgpp = avg;
        Some(pp)
    } else {
        None
    };
    (tr, ppstr, fsc, avgpp)
}

// cm_StochasticParsetreeHB (cm_parsetree.c:3018): sample a parsetree from a
// HMM-banded (non-truncated) float Inside matrix. Banded deck access
// alpha[v][jp_v][dp]; EL deck alpha[M][j][d] is non-banded. RNG draw order
// matches C via the shared sample_helper.
fn cm_stochastic_parsetree_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[u8],
    l: i32,
    mx: &HbMx,
    r: &mut EslRandom,
) -> (Parsetree, f32) {
    let m = cm.m; // EL deck index
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let alpha = &mx.dp;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;

    let mut tr = Parsetree::new(100);
    tr.add_node(1, l, 0, -1, -1, -1);

    let mut pda: Vec<i32> = Vec::new();
    let mut v: i32 = 0;
    let mut j: i32 = l;
    let mut d: i32 = l;
    let mut i: i32 = 1;
    let mut fsc: f32 = 0.0;

    loop {
        let is_el = v == m;
        let stt = if is_el { EL_ST } else { cm.sttype[v as usize] as i32 };

        if !is_el && stt == B_ST {
            let y = cm.cfirst[v as usize];
            let z = cm.cnum[v as usize];
            let jp_y = j - jmin[y as usize];
            let jp_z = j - jmin[z as usize];
            let kmin = (j - jmax[y as usize]).max(hdmin[z as usize][jp_z as usize]);
            let kmax = jp_y.min(hdmax[z as usize][jp_z as usize]);
            let cur = (d + 1) as usize;
            let mut pa = vec![IMPOSSIBLE; cur];
            let mut k = kmin;
            while k <= kmax {
                let jp_yk = (jp_y - k) as usize;
                if k >= d - hdmax[y as usize][jp_yk] && k <= d - hdmin[y as usize][jp_yk] {
                    let kp_z = k - hdmin[z as usize][jp_z as usize];
                    let dp_y = d - hdmin[y as usize][jp_yk];
                    pa[k as usize] = alpha[y as usize][jp_yk][(dp_y - k) as usize]
                        + alpha[z as usize][jp_z as usize][kp_z as usize];
                }
                k += 1;
            }
            let choice =
                sample_helper(r, &mut pa).expect("cm_StochasticParsetreeHB: no valid B_st k");
            let k = choice as i32;
            pda.push(j);
            pda.push(k);
            pda.push(tr.n - 1);
            j -= k;
            d -= k;
            i = j - d + 1;
            let parent = tr.n - 1;
            let idx = tr.add_node(i, j, y, -1, -1, parent);
            tr.nxtl[parent as usize] = idx;
            v = y;
        } else if is_el || stt == E_ST {
            if pda.is_empty() {
                break;
            }
            let bifparent = pda.pop().unwrap();
            d = pda.pop().unwrap();
            j = pda.pop().unwrap();
            v = tr.state[bifparent as usize];
            let y = cm.cnum[v as usize];
            i = j - d + 1;
            let idx = tr.add_node(i, j, y, -1, -1, bifparent);
            tr.nxtr[bifparent as usize] = idx;
            v = y;
        } else {
            let mut yoffset: i32;
            let mut b: i32 = -1;
            if v > 0 || !local_begin {
                fsc += get_femission_score(cm, dsq, v as usize, i, j);
                let sd = state_delta(stt);
                let sdr = state_right_delta(stt);
                let cnum = cm.cnum[v as usize];
                let mut el_is_possible = false;
                let mut cur = cnum as usize;
                if local_end && not_impossible(cm.endsc[v as usize]) {
                    el_is_possible = true;
                    cur += 1;
                }
                let mut pa = vec![IMPOSSIBLE; cur];
                for yo in 0..cnum {
                    let y = cm.cfirst[v as usize] + yo;
                    if (j - sdr) >= jmin[y as usize] && (j - sdr) <= jmax[y as usize] {
                        let jp_y_sdr = (j - jmin[y as usize] - sdr) as usize;
                        if (d - sd) >= hdmin[y as usize][jp_y_sdr]
                            && (d - sd) <= hdmax[y as usize][jp_y_sdr]
                        {
                            let dp_y_sd = (d - hdmin[y as usize][jp_y_sdr] - sd) as usize;
                            pa[yo as usize] = cm.tsc[v as usize][yo as usize]
                                + alpha[y as usize][jp_y_sdr][dp_y_sd];
                        }
                    }
                }
                if el_is_possible {
                    pa[cur - 1] =
                        cm.endsc[v as usize] + alpha[m as usize][j as usize][d as usize];
                }
                let choice = sample_helper(r, &mut pa)
                    .expect("cm_StochasticParsetreeHB: no valid non-B_st transition");
                yoffset = choice as i32;
                if yoffset < cnum {
                    fsc += cm.tsc[v as usize][yoffset as usize];
                } else {
                    yoffset = USED_EL;
                    fsc += cm.endsc[v as usize] + cm.el_selfsc * (d - sd) as f32;
                }
            } else {
                // v == 0 && local begins on: sample the local begin state.
                let cur = m as usize;
                let mut pa = vec![IMPOSSIBLE; cur];
                for y in 0..m {
                    if not_impossible(cm.beginsc[y as usize])
                        && j >= jmin[y as usize]
                        && j <= jmax[y as usize]
                    {
                        let jp_y = (j - jmin[y as usize]) as usize;
                        if d >= hdmin[y as usize][jp_y] && d <= hdmax[y as usize][jp_y] {
                            let dp_y = (d - hdmin[y as usize][jp_y]) as usize;
                            pa[y as usize] =
                                cm.beginsc[y as usize] + alpha[y as usize][jp_y][dp_y];
                        }
                    }
                }
                let choice = sample_helper(r, &mut pa)
                    .expect("cm_StochasticParsetreeHB: no valid local begin");
                b = choice as i32;
                fsc += cm.beginsc[b as usize];
                yoffset = USED_LOCAL_BEGIN;
            }

            match stt {
                x if x == D_ST => {}
                x if x == MP_ST => {
                    i += 1;
                    j -= 1;
                }
                x if x == ML_ST => i += 1,
                x if x == MR_ST => j -= 1,
                x if x == IL_ST => i += 1,
                x if x == IR_ST => j -= 1,
                x if x == S_ST => {}
                _ => panic!("cm_StochasticParsetreeHB: inconceivable state type {}", stt),
            }
            d = j - i + 1;
            let parent = tr.n - 1;
            if yoffset == USED_EL {
                let idx = tr.add_node(i, j, m, -1, -1, parent);
                tr.nxtl[parent as usize] = idx;
                v = m;
            } else if yoffset == USED_LOCAL_BEGIN {
                let idx = tr.add_node(i, j, b, -1, -1, parent);
                tr.nxtl[parent as usize] = idx;
                v = b;
            } else {
                let y = cm.cfirst[v as usize] + yoffset;
                let idx = tr.add_node(i, j, y, -1, -1, parent);
                tr.nxtl[parent as usize] = idx;
                v = y;
            }
        }
    }
    (tr, fsc)
}

// ============================================================================
// HMM-banded (HB) alignment DP — faithful port of the *HB functions of
// cm_dpalign.c (the DEFAULT, non-truncated cmalign path).
//   cm_alignT_hb                     (cm_dpalign.c:258)  -> cm_align_t_hb()
//   cm_AlignHB                       (cm_dpalign.c:740)  -> cm_align_hb()
//   cm_CYKInsideAlignHB              (cm_dpalign.c:1102) -> cm_cyk_inside_align_hb()
//   cm_InsideAlignHB                 (cm_dpalign.c:1777) -> cm_inside_align_hb()
//   cm_OptAccAlignHB                 (cm_dpalign.c:2508) -> cm_optacc_align_hb()
//   cm_OutsideAlignHB                (cm_dpalign.c:4311) -> cm_outside_align_hb()
//   cm_PosteriorHB                   (cm_dpalign.c:4894) -> cm_posterior_hb()
//   cm_EmitterPosteriorHB            (cm_dpalign.c:5194) -> cm_emitter_posterior_hb()
//   cm_PostCodeHB                    (cm_dpalign.c:5509) -> cm_postcode_hb()
//   cm_InitializeOptAccShadowDZeroHB (cm_dpalign.c:5732) -> cm_init_optacc_shadow_dzero_hb()
//
// Banded matrix (C CM_HB_MX): dp[v][jp][dp] for v in 0..M, jp = j-jmin[v],
// dp = d-hdmin[v][jp]. Deck M (EL) is non-banded: dp[M][j][d], j,d in 0..=L.
// Unreachable state decks (jmax[v] < jmin[v]) hold zero rows. Only the
// non-truncated path is ported here (Jvalid[v] is TRUE for all v).
// ============================================================================

struct HbMx {
    dp: Vec<Vec<Vec<f32>>>, // 0..M-1 banded; [M] = EL deck [j][d]
}
struct HbShadowMx {
    yshadow: Vec<Vec<Vec<i32>>>, // 0..M-1
    kshadow: Vec<Vec<Vec<i32>>>, // 0..M-1
}
struct HbEmitMx {
    // l_pp[v] (v<M): index ip_v = i-imin[v]; l_pp[M] (EL): index i (0..=L).
    l_pp: Vec<Option<Vec<f32>>>, // 0..=M
    r_pp: Vec<Option<Vec<f32>>>, // 0..=M ; index jp_v = j-jmin[v]
    sum: Vec<f32>,               // 0..=L
}

#[inline]
fn hb_nrows(cp9b: &CP9Bands, v: usize) -> usize {
    if cp9b.jmax[v] >= cp9b.jmin[v] { (cp9b.jmax[v] - cp9b.jmin[v] + 1) as usize } else { 0 }
}
#[inline]
fn hb_width(cp9b: &CP9Bands, v: usize, jp: usize) -> usize {
    let w = cp9b.hdmax[v][jp] - cp9b.hdmin[v][jp] + 1;
    if w > 0 { w as usize } else { 0 }
}
fn hb_deck_f32(cp9b: &CP9Bands, v: usize, init: f32) -> Vec<Vec<f32>> {
    let nr = hb_nrows(cp9b, v);
    (0..nr).map(|jp| vec![init; hb_width(cp9b, v, jp)]).collect()
}
fn hb_deck_i32(cp9b: &CP9Bands, v: usize, init: i32) -> Vec<Vec<i32>> {
    let nr = hb_nrows(cp9b, v);
    (0..nr).map(|jp| vec![init; hb_width(cp9b, v, jp)]).collect()
}

// ============================================================================
// cm_CYKInsideAlignHB (cm_dpalign.c:1102). Returns (shadow, b, sc).
// ============================================================================
fn cm_cyk_inside_align_hb(cm: &CM, cp9b: &CP9Bands, dsq: &[u8], l: i32) -> (HbShadowMx, i32, f32) {
    let m = cm.m as usize;
    let ll = l as usize;
    let kp = ALPHABET_SIZE_P;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let mut b: i32 = -1;
    let mut bsc: f32 = IMPOSSIBLE;
    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    // alpha decks (0..M-1 built as we go), EL deck at M.
    let mut dp: Vec<Vec<Vec<f32>>> = vec![Vec::new(); m + 1];
    let mut yshadow: Vec<Vec<Vec<i32>>> = vec![Vec::new(); m];
    let mut kshadow: Vec<Vec<Vec<i32>>> = vec![Vec::new(); m];

    // el_scA[d] = el_selfsc * d (cm_dpalign.c:1165)
    let mut el_sca = vec![0.0f32; ll + 1];
    for d in 0..=ll { el_sca[d] = cm.el_selfsc * d as f32; }
    // EL deck (cm_dpalign.c:1185)
    let mut el = vec![vec![IMPOSSIBLE; ll + 1]; ll + 1];
    if local_end {
        for j in 0..=ll { for d in 0..=j { el[j][d] = el_sca[d]; } }
    }
    dp[m] = el;

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let endsc_v = cm.endsc[v];

        let mut a = hb_deck_f32(cp9b, v, IMPOSSIBLE);
        let mut ys = hb_deck_i32(cp9b, v, USED_EL);
        let mut ks = hb_deck_i32(cp9b, v, 0);

        // re-initialize J deck for local ends (cm_dpalign.c:1201)
        if not_impossible(endsc_v) {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let (mut d, mut dp_v);
                if hdmin[v][jp_v] >= sd { d = hdmin[v][jp_v]; dp_v = 0usize; }
                else { d = sd; dp_v = (sd - hdmin[v][jp_v]) as usize; }
                while d <= hdmax[v][jp_v] {
                    if d >= sd {
                        a[jp_v][dp_v] = dp[m][(j) as usize][(d - sd) as usize] + endsc_v;
                    }
                    dp_v += 1; d += 1;
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                a[jp_v][0] = 0.0;
            }
        } else if stt == IL_ST || stt == IR_ST {
            let is_il = stt == IL_ST;
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                // valid children (cm_dpalign.c:1245)
                let mut yvalid: Vec<usize> = Vec::new();
                for yo in 0..cnum {
                    let y = cfirst + yo;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] { yvalid.push(yo); }
                }
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    for &yo in &yvalid {
                        let y = cfirst + yo;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            // self-transit (y==v): read the in-progress deck `a`
                            let child = if y == v { a[jp_y_sdr][dp_y_sd] } else { dp[y][jp_y_sdr][dp_y_sd] };
                            let sc = child + tsc_v[yo];
                            if sc > a[jp_v][dp_v] { a[jp_v][dp_v] = sc; ys[jp_v][dp_v] = yo as i32; }
                        }
                    }
                    a[jp_v][dp_v] += if is_il { esc_v[dsq[i as usize] as usize] } else { esc_v[dsq[j as usize] as usize] };
                    if a[jp_v][dp_v] < IMPOSSIBLE { a[jp_v][dp_v] = IMPOSSIBLE; }
                    d += 1;
                }
            }
        } else if stt != B_ST {
            // ML, MP, MR, D, S (cm_dpalign.c:1310): for y { for j { for d } }
            for yo in 0..cnum {
                let y = cfirst + yo;
                let tsc = tsc_v[yo];
                let jn = jmin[v].max(jmin[y] + sdr);
                let jx = jmax[v].min(jmax[y] + sdr);
                let mut jp_v = jn - jmin[v];
                let mut jp_y_sdr = jn - jmin[y] - sdr;
                let mut jj = jn;
                while jj <= jx {
                    let jpv = jp_v as usize;
                    let jpysdr = jp_y_sdr as usize;
                    let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                    let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                    let mut dp_v = dn - hdmin[v][jpv];
                    let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                    let mut d = dn;
                    while d <= dx {
                        let sc = dp[y][jpysdr][dp_y_sd as usize] + tsc;
                        if sc > a[jpv][dp_v as usize] { a[jpv][dp_v as usize] = sc; ys[jpv][dp_v as usize] = yo as i32; }
                        dp_v += 1; dp_y_sd += 1; d += 1;
                    }
                    jp_v += 1; jp_y_sdr += 1; jj += 1;
                }
            }
            // emissions (cm_dpalign.c:1362)
            match stt {
                x if x == ML_ST => {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let mut i = j - hdmin[v][jp_v] + 1;
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            a[jp_v][dp_v] += esc_v[dsq[i as usize] as usize]; i -= 1;
                        }
                    }
                }
                x if x == MR_ST => {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            a[jp_v][dp_v] += esc_v[dsq[j as usize] as usize];
                        }
                    }
                }
                x if x == MP_ST => {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let mut i = j - hdmin[v][jp_v] + 1;
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            let li = dsq[i as usize] as usize;
                            let ri = dsq[j as usize] as usize;
                            a[jp_v][dp_v] += esc_v[li * kp + ri]; i -= 1;
                        }
                    }
                }
                _ => {}
            }
            // floor (cm_dpalign.c:1388)
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                for dp_v in 0..hb_width(cp9b, v, jp_v) {
                    if a[jp_v][dp_v] < IMPOSSIBLE { a[jp_v][dp_v] = IMPOSSIBLE; }
                }
            }
        } else {
            // B_st (cm_dpalign.c:1395)
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let mut kn = (j - jmax[y]).max(hdmin[z][jp_z]);
                kn = kn.max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let jpyk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jpyk] && k <= d - hdmin[y][jpyk] {
                            let kp_z = (k - hdmin[z][jp_z]) as usize;
                            let dp_y = d - hdmin[y][jpyk];
                            let sc = dp[y][jpyk][(dp_y - k) as usize] + dp[z][jp_z][kp_z];
                            if sc > a[jp_v][dp_v] { a[jp_v][dp_v] = sc; ks[jp_v][dp_v] = k; }
                        }
                        k += 1;
                    }
                    d += 1;
                }
            }
        }

        // local begins (cm_dpalign.c:1462)
        if local_begin && l >= jmin[v] && l <= jmax[v] {
            let jp_v = (l - jmin[v]) as usize;
            if l >= hdmin[v][jp_v] && l <= hdmax[v][jp_v] {
                let lp = (l - hdmin[v][jp_v]) as usize;
                if not_impossible(cm.beginsc[v]) && a[jp_v][lp] + cm.beginsc[v] > bsc {
                    b = v as i32;
                    bsc = a[jp_v][lp] + cm.beginsc[v];
                }
            }
        }

        dp[v] = a;
        yshadow[v] = ys;
        kshadow[v] = ks;
    }

    // fold best local begin into root (cm_dpalign.c:1490)
    if not_impossible(bsc) && bsc > dp[0][jp_0][lp_0] {
        dp[0][jp_0][lp_0] = bsc;
        yshadow[0][jp_0][lp_0] = USED_LOCAL_BEGIN;
    }
    let sc = dp[0][jp_0][lp_0];
    (HbShadowMx { yshadow, kshadow }, b, sc)
}

// ============================================================================
// cm_InsideAlignHB (cm_dpalign.c:1777). Returns (HbMx, sc).
// ============================================================================
fn cm_inside_align_hb(cm: &CM, cp9b: &CP9Bands, dsq: &[u8], l: i32) -> (HbMx, f32) {
    let m = cm.m as usize;
    let ll = l as usize;
    let kp = ALPHABET_SIZE_P;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let mut bsc = IMPOSSIBLE;
    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    let mut dp: Vec<Vec<Vec<f32>>> = vec![Vec::new(); m + 1];
    let mut el_sca = vec![0.0f32; ll + 1];
    for d in 0..=ll { el_sca[d] = cm.el_selfsc * d as f32; }
    let mut el = vec![vec![IMPOSSIBLE; ll + 1]; ll + 1];
    if local_end {
        for j in 0..=ll { for d in 0..=j { el[j][d] = el_sca[d]; } }
    }
    dp[m] = el;

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let endsc_v = cm.endsc[v];

        let mut a = hb_deck_f32(cp9b, v, IMPOSSIBLE);

        // re-init J deck for local ends (cm_dpalign.c:1868)
        if not_impossible(endsc_v) {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut d = hdmin[v][jp_v];
                let mut dp_v = 0usize;
                while d <= hdmax[v][jp_v] {
                    a[jp_v][dp_v] = el_sca[(d - sd) as usize] + endsc_v;
                    dp_v += 1; d += 1;
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] { a[(j - jmin[v]) as usize][0] = 0.0; }
        } else if stt == IL_ST || stt == IR_ST {
            let is_il = stt == IL_ST;
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yo in 0..cnum {
                    let y = cfirst + yo;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] { yvalid.push(yo); }
                }
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    for &yo in &yvalid {
                        let y = cfirst + yo;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            // self-transit (y==v): read the in-progress deck `a`
                            let child = if y == v { a[jp_y_sdr][dp_y_sd] } else { dp[y][jp_y_sdr][dp_y_sd] };
                            a[jp_v][dp_v] = flogsum(a[jp_v][dp_v], child + tsc_v[yo]);
                        }
                    }
                    a[jp_v][dp_v] += if is_il { esc_v[dsq[i as usize] as usize] } else { esc_v[dsq[j as usize] as usize] };
                    if a[jp_v][dp_v] < IMPOSSIBLE { a[jp_v][dp_v] = IMPOSSIBLE; }
                    d += 1;
                }
            }
        } else if stt != B_ST {
            for yo in 0..cnum {
                let y = cfirst + yo;
                let tsc = tsc_v[yo];
                let jn = jmin[v].max(jmin[y] + sdr);
                let jx = jmax[v].min(jmax[y] + sdr);
                let mut jp_v = jn - jmin[v];
                let mut jp_y_sdr = jn - jmin[y] - sdr;
                let mut jj = jn;
                while jj <= jx {
                    let jpv = jp_v as usize;
                    let jpysdr = jp_y_sdr as usize;
                    let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                    let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                    let mut dp_v = dn - hdmin[v][jpv];
                    let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                    let mut d = dn;
                    while d <= dx {
                        a[jpv][dp_v as usize] = flogsum(a[jpv][dp_v as usize], dp[y][jpysdr][dp_y_sd as usize] + tsc);
                        dp_v += 1; dp_y_sd += 1; d += 1;
                    }
                    jp_v += 1; jp_y_sdr += 1; jj += 1;
                }
            }
            match stt {
                x if x == ML_ST => {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let mut i = j - hdmin[v][jp_v] + 1;
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            a[jp_v][dp_v] += esc_v[dsq[i as usize] as usize]; i -= 1;
                        }
                    }
                }
                x if x == MR_ST => {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            a[jp_v][dp_v] += esc_v[dsq[j as usize] as usize];
                        }
                    }
                }
                x if x == MP_ST => {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let mut i = j - hdmin[v][jp_v] + 1;
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            let li = dsq[i as usize] as usize;
                            let ri = dsq[j as usize] as usize;
                            a[jp_v][dp_v] += esc_v[li * kp + ri]; i -= 1;
                        }
                    }
                }
                _ => {}
            }
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                for dp_v in 0..hb_width(cp9b, v, jp_v) {
                    if a[jp_v][dp_v] < IMPOSSIBLE { a[jp_v][dp_v] = IMPOSSIBLE; }
                }
            }
        } else {
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let mut kn = (j - jmax[y]).max(hdmin[z][jp_z]);
                kn = kn.max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let jpyk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jpyk] && k <= d - hdmin[y][jpyk] {
                            let kp_z = (k - hdmin[z][jp_z]) as usize;
                            let dp_y = d - hdmin[y][jpyk];
                            a[jp_v][dp_v] = flogsum(a[jp_v][dp_v], dp[y][jpyk][(dp_y - k) as usize] + dp[z][jp_z][kp_z]);
                        }
                        k += 1;
                    }
                    d += 1;
                }
            }
        }

        if local_begin && not_impossible(cm.beginsc[v]) && l >= jmin[v] && l <= jmax[v] {
            let jp_v = (l - jmin[v]) as usize;
            if l >= hdmin[v][jp_v] && l <= hdmax[v][jp_v] {
                let lp = (l - hdmin[v][jp_v]) as usize;
                bsc = flogsum(bsc, a[jp_v][lp] + cm.beginsc[v]);
            }
        }
        dp[v] = a;
    }

    if not_impossible(bsc) {
        dp[0][jp_0][lp_0] = flogsum(dp[0][jp_0][lp_0], bsc);
    }
    let sc = dp[0][jp_0][lp_0];
    (HbMx { dp }, sc)
}

// ============================================================================
// cm_OutsideAlignHB (cm_dpalign.c:4311). do_check omitted (default off). Fills
// beta; returns HbMx.
// ============================================================================
fn cm_outside_align_hb(cm: &CM, cp9b: &CP9Bands, dsq: &[u8], l: i32, ins: &HbMx) -> HbMx {
    let m = cm.m as usize;
    let ll = l as usize;
    let kp = ALPHABET_SIZE_P;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let alpha = &ins.dp;

    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    // beta decks 0..M-1, plus separate EL deck.
    let mut beta: Vec<Vec<Vec<f32>>> = (0..m).map(|v| hb_deck_f32(cp9b, v, IMPOSSIBLE)).collect();
    let mut el = vec![vec![IMPOSSIBLE; ll + 1]; ll + 1];

    beta[0][jp_0][lp_0] = 0.0;
    // local begin cells (cm_dpalign.c:4373)
    if local_begin {
        for v in 1..m {
            if not_impossible(cm.beginsc[v]) && l >= jmin[v] && l <= jmax[v] {
                let jp_v = (l - jmin[v]) as usize;
                if l >= hdmin[v][jp_v] && l <= hdmax[v][jp_v] {
                    let lp = (l - hdmin[v][jp_v]) as usize;
                    beta[v][jp_v][lp] = cm.beginsc[v];
                }
            }
        }
    }

    for v in 1..m {
        let stt = cm.sttype[v] as i32;
        let stid = cm.stid[v] as i32;
        let mut bv = std::mem::take(&mut beta[v]);

        if stid == BEGL_S {
            let y = cm.plast[v] as usize;
            let z = cm.cnum[y] as usize;
            let mut j = jmax[v];
            while j >= jmin[v] {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = j - jmin[z];
                let mut d = hdmax[v][jp_v];
                while d >= hdmin[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let kmin = jmin[y].max(jmin[z]) - j;
                    let kmax = jmax[y].min(jmax[z]) - j;
                    let mut k = kmin;
                    while k <= kmax {
                        let jp_yk = (jp_y + k) as usize;
                        let jp_zk = (jp_z + k) as usize;
                        if k < hdmin[y][jp_yk] - d || k > hdmax[y][jp_yk] - d { k += 1; continue; }
                        if k < hdmin[z][jp_zk] || k > hdmax[z][jp_zk] { k += 1; continue; }
                        let kp_z = (k - hdmin[z][jp_zk]) as usize;
                        let dp_y = d - hdmin[y][jp_yk];
                        bv[jp_v][dp_v] = flogsum(bv[jp_v][dp_v], beta[y][jp_yk][(dp_y + k) as usize] + alpha[z][jp_zk][kp_z]);
                        k += 1;
                    }
                    d -= 1;
                }
                j -= 1;
            }
        } else if stid == BEGR_S {
            let y = cm.plast[v] as usize;
            let z = cm.cfirst[y] as usize;
            let jn = jmin[v].max(jmin[y]);
            let jx = jmax[v].min(jmax[y]);
            let mut j = jx;
            while j >= jn {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = j - jmin[z];
                let dn = hdmin[v][jp_v].max(j - jmax[z]);
                let dx = hdmax[v][jp_v].min(jp_z);
                let mut d = dx;
                while d >= dn {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let jp_zd = (jp_z - d) as usize;
                    let kmin = (hdmin[y][jp_y as usize] - d).max(hdmin[z][jp_zd]);
                    let kmax = (hdmax[y][jp_y as usize] - d).min(hdmax[z][jp_zd]);
                    let mut k = kmin;
                    while k <= kmax {
                        let kp_z = (k - hdmin[z][jp_zd]) as usize;
                        let dp_y = d - hdmin[y][jp_y as usize];
                        bv[jp_v][dp_v] = flogsum(bv[jp_v][dp_v], beta[y][jp_y as usize][(dp_y + k) as usize] + alpha[z][jp_zd][kp_z]);
                        k += 1;
                    }
                    d -= 1;
                }
                j -= 1;
            }
        } else if stt == IL_ST || stt == IR_ST {
            let mut j = jmax[v];
            while j >= jmin[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut d = hdmax[v][jp_v];
                while d >= hdmin[v][jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let plast = cm.plast[v];
                    let pnum = cm.pnum[v];
                    let mut y = plast;
                    while y > plast - pnum {
                        let yu = y as usize;
                        let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                        let yst = cm.sttype[yu] as i32;
                        match yst {
                            x if x == MP_ST => {
                                if !(j == l || d == j)
                                    && (j + 1) >= jmin[yu] && (j + 1) <= jmax[yu] {
                                    let jp_y = j - jmin[yu];
                                    if (d + 2) >= hdmin[yu][(jp_y + 1) as usize] && (d + 2) <= hdmax[yu][(jp_y + 1) as usize] {
                                        let dp_y = d - hdmin[yu][(jp_y + 1) as usize];
                                        let escore = cm.oesc[yu][dsq[(i - 1) as usize] as usize * kp + dsq[(j + 1) as usize] as usize];
                                        bv[jp_v][dp_v] = flogsum(bv[jp_v][dp_v], beta[yu][(jp_y + 1) as usize][(dp_y + 2) as usize] + cm.tsc[yu][voffset] + escore);
                                    }
                                }
                            }
                            x if x == ML_ST || x == IL_ST => {
                                if d != j && j >= jmin[yu] && j <= jmax[yu] {
                                    let jp_y = j - jmin[yu];
                                    if (d + 1) >= hdmin[yu][jp_y as usize] && (d + 1) <= hdmax[yu][jp_y as usize] {
                                        let dp_y = d - hdmin[yu][jp_y as usize];
                                        let escore = cm.oesc[yu][dsq[(i - 1) as usize] as usize];
                                        // self-transit (yu==v): read the in-progress deck `bv`
                                        let other = if yu == v { bv[jp_y as usize][(dp_y + 1) as usize] } else { beta[yu][jp_y as usize][(dp_y + 1) as usize] };
                                        bv[jp_v][dp_v] = flogsum(bv[jp_v][dp_v], other + cm.tsc[yu][voffset] + escore);
                                    }
                                }
                            }
                            x if x == MR_ST || x == IR_ST => {
                                if j != l && (j + 1) >= jmin[yu] && (j + 1) <= jmax[yu] {
                                    let jp_y = j - jmin[yu];
                                    if (d + 1) >= hdmin[yu][(jp_y + 1) as usize] && (d + 1) <= hdmax[yu][(jp_y + 1) as usize] {
                                        let dp_y = d - hdmin[yu][(jp_y + 1) as usize];
                                        let escore = cm.oesc[yu][dsq[(j + 1) as usize] as usize];
                                        // self-transit (yu==v): read the in-progress deck `bv`
                                        let other = if yu == v { bv[(jp_y + 1) as usize][(dp_y + 1) as usize] } else { beta[yu][(jp_y + 1) as usize][(dp_y + 1) as usize] };
                                        bv[jp_v][dp_v] = flogsum(bv[jp_v][dp_v], other + cm.tsc[yu][voffset] + escore);
                                    }
                                }
                            }
                            x if x == S_ST || x == E_ST || x == D_ST => {
                                if j >= jmin[yu] && j <= jmax[yu] {
                                    let jp_y = j - jmin[yu];
                                    if d >= hdmin[yu][jp_y as usize] && d <= hdmax[yu][jp_y as usize] {
                                        let dp_y = d - hdmin[yu][jp_y as usize];
                                        bv[jp_v][dp_v] = flogsum(bv[jp_v][dp_v], beta[yu][jp_y as usize][dp_y as usize] + cm.tsc[yu][voffset]);
                                    }
                                }
                            }
                            _ => {}
                        }
                        y -= 1;
                    }
                    if bv[jp_v][dp_v] < IMPOSSIBLE { bv[jp_v][dp_v] = IMPOSSIBLE; }
                    d -= 1;
                }
                j -= 1;
            }
        } else {
            // ML, MP, MR, D, S, B, E (cm_dpalign.c:4579): for y (parents) { for j { for d } }
            let plast = cm.plast[v];
            let pnum = cm.pnum[v];
            let mut y = plast;
            while y > plast - pnum {
                let yu = y as usize;
                let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                let yst = cm.sttype[yu] as i32;
                let sdr = state_right_delta(yst);
                let sd = state_delta(yst);
                let jn = jmin[v].max(jmin[yu] - sdr);
                let jx = jmax[v].min(jmax[yu] - sdr);
                let mut j = jx;
                while j >= jn {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_y = j - jmin[yu];
                    let dn = hdmin[v][jp_v].max(hdmin[yu][(jp_y + sdr) as usize] - sd);
                    let dx = hdmax[v][jp_v].min(hdmax[yu][(jp_y + sdr) as usize] - sd);
                    let mut dp_v = (dx - hdmin[v][jp_v]) as i32;
                    let mut dp_y = (dx - hdmin[yu][(jp_y + sdr) as usize]) as i32;
                    let mut i = j - dx + 1;
                    let jp_ysdr = (jp_y + sdr) as usize;
                    match yst {
                        x if x == MP_ST => {
                            let mut d = dx;
                            while d >= dn {
                                let escore = cm.oesc[yu][dsq[(i - 1) as usize] as usize * kp + dsq[(j + 1) as usize] as usize];
                                bv[jp_v][dp_v as usize] = flogsum(bv[jp_v][dp_v as usize], beta[yu][jp_ysdr][(dp_y + sd) as usize] + cm.tsc[yu][voffset] + escore);
                                d -= 1; dp_v -= 1; dp_y -= 1; i += 1;
                            }
                        }
                        x if x == ML_ST || x == IL_ST => {
                            let mut d = dx;
                            while d >= dn {
                                let escore = cm.oesc[yu][dsq[(i - 1) as usize] as usize];
                                bv[jp_v][dp_v as usize] = flogsum(bv[jp_v][dp_v as usize], beta[yu][jp_ysdr][(dp_y + sd) as usize] + cm.tsc[yu][voffset] + escore);
                                d -= 1; dp_v -= 1; dp_y -= 1; i += 1;
                            }
                        }
                        x if x == MR_ST || x == IR_ST => {
                            let escore = cm.oesc[yu][dsq[(j + 1) as usize] as usize];
                            let mut d = dx;
                            while d >= dn {
                                bv[jp_v][dp_v as usize] = flogsum(bv[jp_v][dp_v as usize], beta[yu][jp_ysdr][(dp_y + sd) as usize] + cm.tsc[yu][voffset] + escore);
                                d -= 1; dp_v -= 1; dp_y -= 1;
                            }
                        }
                        _ => {
                            // EMITNONE: D, S, E
                            let mut d = dx;
                            while d >= dn {
                                bv[jp_v][dp_v as usize] = flogsum(bv[jp_v][dp_v as usize], beta[yu][jp_ysdr][(dp_y + sd) as usize] + cm.tsc[yu][voffset]);
                                d -= 1; dp_v -= 1; dp_y -= 1;
                            }
                        }
                    }
                    j -= 1;
                }
                y -= 1;
            }
        }

        // v -> EL (cm_dpalign.c:4650)
        if local_end && not_impossible(cm.endsc[v]) {
            let sdr = state_right_delta(stt);
            let sd = state_delta(stt);
            let jn = jmin[v] - sdr;
            let jx = jmax[v] - sdr;
            let mut j = jn;
            while j <= jx {
                let jp_v = j - jmin[v]; // may be negative index base; offset by +sdr below
                let dn = hdmin[v][(jp_v + sdr) as usize] - sd;
                let dx = hdmax[v][(jp_v + sdr) as usize] - sd;
                let mut i = j - dn + 1;
                let mut dpv = dn - hdmin[v][(jp_v + sdr) as usize];
                let jp_vsdr = (jp_v + sdr) as usize;
                match stt {
                    x if x == MP_ST => {
                        let mut d = dn;
                        while d <= dx {
                            let escore = cm.oesc[v][dsq[(i - 1) as usize] as usize * kp + dsq[(j + 1) as usize] as usize];
                            el[j as usize][d as usize] = flogsum(el[j as usize][d as usize], bv[jp_vsdr][(dpv + sd) as usize] + cm.endsc[v] + escore);
                            d += 1; dpv += 1; i -= 1;
                        }
                    }
                    x if x == ML_ST || x == IL_ST => {
                        let mut d = dn;
                        while d <= dx {
                            let escore = cm.oesc[v][dsq[(i - 1) as usize] as usize];
                            el[j as usize][d as usize] = flogsum(el[j as usize][d as usize], bv[jp_vsdr][(dpv + sd) as usize] + cm.endsc[v] + escore);
                            d += 1; dpv += 1; i -= 1;
                        }
                    }
                    x if x == MR_ST || x == IR_ST => {
                        let escore = cm.oesc[v][dsq[(j + 1) as usize] as usize];
                        let mut d = dn;
                        while d <= dx {
                            el[j as usize][d as usize] = flogsum(el[j as usize][d as usize], bv[jp_vsdr][(dpv + sd) as usize] + cm.endsc[v] + escore);
                            d += 1; dpv += 1;
                        }
                    }
                    _ => {
                        let mut d = dn;
                        while d <= dx {
                            el[j as usize][d as usize] = flogsum(el[j as usize][d as usize], bv[jp_vsdr][(dpv + sd) as usize] + cm.endsc[v]);
                            d += 1; dpv += 1;
                        }
                    }
                }
                j += 1;
            }
        }

        beta[v] = bv;
    }

    // EL->EL left-emitting (cm_dpalign.c:4701)
    if local_end {
        let mut j = l;
        while j > 0 {
            let mut d = j - 1;
            while d >= 0 {
                el[j as usize][d as usize] = flogsum(el[j as usize][d as usize], el[j as usize][(d + 1) as usize] + cm.el_selfsc);
                d -= 1;
            }
            j -= 1;
        }
    }

    beta.push(el); // deck M
    HbMx { dp: beta }
}

// ============================================================================
// cm_PosteriorHB (cm_dpalign.c:4894). Returns post as HbMx.
// ============================================================================
fn cm_posterior_hb(cm: &CM, cp9b: &CP9Bands, l: i32, ins: &HbMx, out: &HbMx) -> HbMx {
    let m = cm.m as usize;
    let ll = l as usize;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;
    let sc = ins.dp[0][jp_0][lp_0];

    let mut post: Vec<Vec<Vec<f32>>> = (0..m).map(|v| hb_deck_f32(cp9b, v, IMPOSSIBLE)).collect();
    let mut el = vec![vec![IMPOSSIBLE; ll + 1]; ll + 1];
    if local_end {
        for j in 0..=ll {
            for d in 0..=j {
                el[j][d] = ins.dp[m][j][d] + out.dp[m][j][d] - sc;
            }
        }
    }
    for v in 0..m {
        for j in jmin[v]..=jmax[v] {
            let jp_v = (j - jmin[v]) as usize;
            for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                let dp_v = (d - hdmin[v][jp_v]) as usize;
                post[v][jp_v][dp_v] = ins.dp[v][jp_v][dp_v] + out.dp[v][jp_v][dp_v] - sc;
            }
        }
    }
    post.push(el);
    HbMx { dp: post }
}

// ============================================================================
// cm_EmitterPosteriorHB (cm_dpalign.c:5194). do_check omitted (default off).
// ============================================================================
fn cm_emitter_posterior_hb(cm: &CM, cp9b: &CP9Bands, l: i32, post: &HbMx) -> HbEmitMx {
    let m = cm.m as usize;
    let ll = l as usize;
    let imin = &cp9b.imin;
    let imax = &cp9b.imax;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let ilen = |v: usize| -> usize { if imax[v] >= imin[v] { (imax[v] - imin[v] + 1) as usize } else { 0 } };
    let jlen = |v: usize| -> usize { if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 } };

    let mut l_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut r_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            l_pp[v] = Some(vec![IMPOSSIBLE; ilen(v)]);
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            r_pp[v] = Some(vec![IMPOSSIBLE; jlen(v)]);
        }
    }
    if local_end {
        l_pp[m] = Some(vec![IMPOSSIBLE; ll + 1]);
    }

    // Step 1 (cm_dpalign.c:5226)
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            let lp = l_pp[v].as_mut().unwrap();
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let i = j - d + 1;
                    let ip_v = (i - imin[v]) as usize;
                    lp[ip_v] = flogsum(lp[ip_v], post.dp[v][jp_v][dp_v]);
                }
            }
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            let rp = r_pp[v].as_mut().unwrap();
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    rp[jp_v] = flogsum(rp[jp_v], post.dp[v][jp_v][dp_v]);
                }
            }
        }
    }
    // EL contribution (cm_dpalign.c:5252), non-banded EL deck
    if local_end {
        let lp = l_pp[m].as_mut().unwrap();
        for j in 1..=l {
            let mut i = j;
            for d in 1..=j {
                lp[i as usize] = flogsum(lp[i as usize], post.dp[m][j as usize][d as usize]);
                i -= 1;
            }
        }
    }

    // Step 2: normalize (cm_dpalign.c:5270)
    let mut sum = vec![IMPOSSIBLE; ll + 1];
    for v in 0..m {
        if let Some(lp) = &l_pp[v] {
            for i in imin[v]..=imax[v] {
                let ip_v = (i - imin[v]) as usize;
                sum[i as usize] = flogsum(sum[i as usize], lp[ip_v]);
            }
        }
        if let Some(rp) = &r_pp[v] {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                sum[j as usize] = flogsum(sum[j as usize], rp[jp_v]);
            }
        }
    }
    if let Some(lp) = &l_pp[m] {
        for i in 1..=ll { sum[i] = flogsum(sum[i], lp[i]); }
    }
    // normalize
    for v in 0..m {
        if let Some(lp) = l_pp[v].as_mut() {
            for i in imin[v]..=imax[v] {
                let ip_v = (i - imin[v]) as usize;
                lp[ip_v] -= sum[i as usize];
            }
        }
        if let Some(rp) = r_pp[v].as_mut() {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                rp[jp_v] -= sum[j as usize];
            }
        }
    }
    if let Some(lp) = l_pp[m].as_mut() {
        for i in 1..=ll { lp[i] -= sum[i]; }
    }

    // Step 3: combine MATP_MP (v) with MATP_ML (v+1) and MATP_MR (v+2)
    // (cm_dpalign.c:5333)
    for v in 0..m {
        if cm.sttype[v] as i32 == MP_ST {
            // left: combine v and v+1 over shared i band
            if imax[v] >= 1 && imax[v + 1] >= 1 {
                let in_ = imin[v].max(imin[v + 1]);
                let ix = imax[v].min(imax[v + 1]);
                for i in in_..=ix {
                    let ip_v = (i - imin[v]) as usize;
                    let ip_v2 = (i - imin[v + 1]) as usize;
                    let a = l_pp[v].as_ref().unwrap()[ip_v];
                    let bb = l_pp[v + 1].as_ref().unwrap()[ip_v2];
                    let s = flogsum(a, bb);
                    l_pp[v].as_mut().unwrap()[ip_v] = s;
                    l_pp[v + 1].as_mut().unwrap()[ip_v2] = s;
                }
            }
            // right: combine v and v+2 over shared j band
            if jmax[v] >= 1 && jmax[v + 2] >= 1 {
                let jn = jmin[v].max(jmin[v + 2]);
                let jx = jmax[v].min(jmax[v + 2]);
                for j in jn..=jx {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_v2 = (j - jmin[v + 2]) as usize;
                    let a = r_pp[v].as_ref().unwrap()[jp_v];
                    let bb = r_pp[v + 2].as_ref().unwrap()[jp_v2];
                    let s = flogsum(a, bb);
                    r_pp[v].as_mut().unwrap()[jp_v] = s;
                    r_pp[v + 2].as_mut().unwrap()[jp_v2] = s;
                }
            }
        }
    }

    HbEmitMx { l_pp, r_pp, sum }
}

// ============================================================================
// cm_InitializeOptAccShadowDZeroHB (cm_dpalign.c:5732). Jvalid[v]==TRUE for all
// v in the non-truncated path.
// ============================================================================
pub(crate) fn cm_init_optacc_shadow_dzero_hb(cm: &CM, cp9b: &CP9Bands, yshadow: &mut [Vec<Vec<i32>>], l: i32) {
    let m = cm.m as usize;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let have_el = cm.flags & CMH_LOCAL_END != 0;
    let mut esc: Vec<f32> = vec![IMPOSSIBLE; m];
    let endsc: f32;
    if have_el {
        let mut v0 = 0usize;
        while !not_impossible(cm.endsc[v0]) { v0 += 1; }
        endsc = cm.endsc[v0];
    } else {
        endsc = IMPOSSIBLE;
    }

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        if stt == E_ST {
            if have_el { esc[v] = 0.0; }
        } else if stt == B_ST {
            if have_el {
                let y = cm.cfirst[v] as usize;
                let z = cm.cnum[v] as usize;
                esc[v] = esc[y] + esc[z];
            }
        } else {
            let mut y = cm.cfirst[v] as usize;
            while state_delta(cm.sttype[y] as i32) != 0 { y += 1; }
            let mut yoffset = (y as i32) - cm.cfirst[v];
            if have_el {
                esc[v] = esc[y] + cm.tsc[v][yoffset as usize];
                if endsc > esc[v] { yoffset = USED_EL; }
            }
            let jstart = sd.max(jmin[v]);
            for j in jstart..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                if hdmin[v][jp_v] <= hdmax[v][jp_v]
                    && (j - sdr) >= jmin[y] && (j - sdr) <= jmax[y] {
                    let jp_y = (j - sdr - jmin[y]) as usize;
                    if sd >= hdmin[v][jp_v] && sd <= hdmax[v][jp_v]
                        && 0 >= hdmin[y][jp_y] && 0 <= hdmax[y][jp_y] {
                        let dp_v = (sd - hdmin[v][jp_v]) as usize;
                        yshadow[v][jp_v][dp_v] = yoffset;
                    }
                }
            }
        }
    }
}

// ============================================================================
// cm_OptAccAlignHB (cm_dpalign.c:2508). Returns (shadow, b, pp).
// ============================================================================
fn cm_optacc_align_hb(cm: &CM, cp9b: &CP9Bands, l: i32, emit_mx: &HbEmitMx) -> (HbShadowMx, i32, f32) {
    let m = cm.m as usize;
    let ll = l as usize;
    let imin = &cp9b.imin;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let have_el = cm.flags & CMH_LOCAL_END != 0;

    let mut b: i32 = -1;
    let mut bsc = IMPOSSIBLE;
    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    let mut dp: Vec<Vec<Vec<f32>>> = vec![Vec::new(); m + 1];
    let mut yshadow: Vec<Vec<Vec<i32>>> = (0..m).map(|v| hb_deck_i32(cp9b, v, USED_EL)).collect();
    let mut kshadow: Vec<Vec<Vec<i32>>> = vec![Vec::new(); m];

    // d==0 shadow init (cm_dpalign.c:2586)
    cm_init_optacc_shadow_dzero_hb(cm, cp9b, &mut yshadow, l);

    // EL deck (non-banded) (cm_dpalign.c:2589)
    let mut el = vec![vec![IMPOSSIBLE; ll + 1]; ll + 1];
    if have_el {
        if let Some(lpm) = &emit_mx.l_pp[m] {
            for j in 0..=ll {
                el[j][0] = lpm[0];
                let mut i = j;
                for d in 1..=j {
                    el[j][d] = flogsum(el[j][d - 1], lpm[i]);
                    i -= 1;
                }
            }
        }
    }
    dp[m] = el;

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let endsc_v = cm.endsc[v];

        let mut a = hb_deck_f32(cp9b, v, IMPOSSIBLE);
        let mut ys = std::mem::take(&mut yshadow[v]); // keep the d==0 init
        let mut ks = hb_deck_i32(cp9b, v, 0);

        // re-init from EL deck (cm_dpalign.c:2611)
        if have_el && not_impossible(endsc_v) {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    a[jp_v][dp_v] = dp[m][(j - sdr) as usize][(d - sd) as usize];
                }
            }
        }

        if stt == IL_ST || stt == IR_ST {
            let is_il = stt == IL_ST;
            let lp = if is_il { emit_mx.l_pp[v].as_ref() } else { None };
            let rp = if is_il { None } else { emit_mx.r_pp[v].as_ref() };
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yo in 0..cnum {
                    let y = cfirst + yo;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] { yvalid.push(yo); }
                }
                let mut i = j - hdmin[v][jp_v] + 1;
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let ip_v = (i - imin[v]) as usize;
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    for &yo in &yvalid {
                        let y = cfirst + yo;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            // self-transit (y==v): read the in-progress deck `a`
                            let sc = if y == v { a[jp_y_sdr][dp_y_sd] } else { dp[y][jp_y_sdr][dp_y_sd] };
                            if sc > a[jp_v][dp_v] { a[jp_v][dp_v] = sc; ys[jp_v][dp_v] = yo as i32; }
                        }
                    }
                    if is_il { a[jp_v][dp_v] = flogsum(a[jp_v][dp_v], lp.unwrap()[ip_v]); }
                    else { a[jp_v][dp_v] = flogsum(a[jp_v][dp_v], rp.unwrap()[jp_v]); }
                    if a[jp_v][dp_v] < IMPOSSIBLE { a[jp_v][dp_v] = IMPOSSIBLE; }
                    if !have_el && ys[jp_v][dp_v] == USED_EL && d > sd { a[jp_v][dp_v] = IMPOSSIBLE; }
                    i -= 1; d += 1;
                }
            }
        } else if stt != B_ST {
            // ML, MP, MR, D, S (cm_dpalign.c:2675)
            for yo in 0..cnum {
                let y = cfirst + yo;
                let jn = jmin[v].max(jmin[y] + sdr);
                let jx = jmax[v].min(jmax[y] + sdr);
                let mut jp_v = jn - jmin[v];
                let mut jp_y_sdr = jn - jmin[y] - sdr;
                let mut jj = jn;
                while jj <= jx {
                    let jpv = jp_v as usize;
                    let jpysdr = jp_y_sdr as usize;
                    let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                    let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                    let mut dp_v = dn - hdmin[v][jpv];
                    let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                    let mut d = dn;
                    while d <= dx {
                        let sc = dp[y][jpysdr][dp_y_sd as usize];
                        if sc > a[jpv][dp_v as usize] { a[jpv][dp_v as usize] = sc; ys[jpv][dp_v as usize] = yo as i32; }
                        dp_v += 1; dp_y_sd += 1; d += 1;
                    }
                    jp_v += 1; jp_y_sdr += 1; jj += 1;
                }
            }
            // emissions (cm_dpalign.c:2711)
            match stt {
                x if x == ML_ST => {
                    let lp = emit_mx.l_pp[v].as_ref().unwrap();
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let i = j - hdmin[v][jp_v] + 1;
                        let mut ip_v = i - imin[v];
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            a[jp_v][dp_v] = flogsum(a[jp_v][dp_v], lp[ip_v as usize]); ip_v -= 1;
                        }
                    }
                }
                x if x == MR_ST => {
                    let rp = emit_mx.r_pp[v].as_ref().unwrap();
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            a[jp_v][dp_v] = flogsum(a[jp_v][dp_v], rp[jp_v]);
                        }
                    }
                }
                x if x == MP_ST => {
                    let lp = emit_mx.l_pp[v].as_ref().unwrap();
                    let rp = emit_mx.r_pp[v].as_ref().unwrap();
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let i = j - hdmin[v][jp_v] + 1;
                        let mut ip_v = i - imin[v];
                        for dp_v in 0..hb_width(cp9b, v, jp_v) {
                            a[jp_v][dp_v] = flogsum(a[jp_v][dp_v], flogsum(lp[ip_v as usize], rp[jp_v])); ip_v -= 1;
                        }
                    }
                }
                _ => {}
            }
            // floor
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                for dp_v in 0..hb_width(cp9b, v, jp_v) {
                    if a[jp_v][dp_v] < IMPOSSIBLE { a[jp_v][dp_v] = IMPOSSIBLE; }
                }
            }
            // disallow EL requiring emissions if !have_el (cm_dpalign.c:2758)
            if !have_el && sd > 0 {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut d = (sd + 1).max(hdmin[v][jp_v]);
                    let mut dp_v = (d - hdmin[v][jp_v]) as usize;
                    while d <= hdmax[v][jp_v] {
                        if ys[jp_v][dp_v] == USED_EL { a[jp_v][dp_v] = IMPOSSIBLE; }
                        dp_v += 1; d += 1;
                    }
                }
            }
        } else {
            // B_st (cm_dpalign.c:2770)
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let mut kn = (j - jmax[y]).max(hdmin[z][jp_z]);
                kn = kn.max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let jpyk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jpyk] && k <= d - hdmin[y][jpyk] {
                            let kp_z = (k - hdmin[z][jp_z]) as usize;
                            let dp_y = d - hdmin[y][jpyk];
                            let lft = dp[y][jpyk][(dp_y - k) as usize];
                            let rgt = dp[z][jp_z][kp_z];
                            let sc = flogsum(lft, rgt);
                            if sc > a[jp_v][dp_v]
                                && ((d == k) || not_impossible(lft))
                                && ((k == 0) || not_impossible(rgt)) {
                                a[jp_v][dp_v] = sc; ks[jp_v][dp_v] = k;
                            }
                        }
                        k += 1;
                    }
                    d += 1;
                }
            }
        }

        // local begins (cm_dpalign.c:2879)
        if local_begin && not_impossible(cm.beginsc[v]) && l >= jmin[v] && l <= jmax[v] {
            let jp_v = (l - jmin[v]) as usize;
            if l >= hdmin[v][jp_v] && l <= hdmax[v][jp_v] {
                let lp = (l - hdmin[v][jp_v]) as usize;
                if a[jp_v][lp] > bsc { b = v as i32; bsc = a[jp_v][lp]; }
            }
        }

        dp[v] = a;
        yshadow[v] = ys;
        kshadow[v] = ks;
    }

    // fold local begin (cm_dpalign.c:2914)
    if not_impossible(bsc) && local_begin {
        dp[0][jp_0][lp_0] = bsc;
        yshadow[0][jp_0][lp_0] = USED_LOCAL_BEGIN;
    }
    let sc = dp[0][jp_0][lp_0];
    let pp = (sre_exp2(sc) / l as f64) as f32;
    (HbShadowMx { yshadow, kshadow }, b, pp)
}

// ============================================================================
// cm_alignT_hb (cm_dpalign.c:258) — banded traceback into a Parsetree.
// ============================================================================
fn cm_align_t_hb(cm: &CM, cp9b: &CP9Bands, l: i32, sh: &HbShadowMx, b: i32, do_optacc: bool) -> Parsetree {
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let mut tr = Parsetree::new(100);
    tr.add_node(1, l, 0, -1, -1, -1);
    let mut pda: Vec<(i32, i32, i32)> = Vec::new(); // (j, k, bifparent)
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = l;
    let mut d: i32 = l;

    loop {
        let vu = v as usize;
        // special HB optacc BEGL_S/BEGR_S d==0 out-of-band case (cm_dpalign.c:299)
        let mut allow_s_local_end = false;
        let mut jp_v = 0usize;
        let mut dp_v = 0usize;
        if v != cm.m
            && do_optacc && d == 0
            && (cm.stid[vu] as i32 == BEGL_S || cm.stid[vu] as i32 == BEGR_S)
            && ((j < jmin[vu] || j > jmax[vu])
                || (d < hdmin[vu][(j - jmin[vu]) as usize] || d > hdmax[vu][(j - jmin[vu]) as usize]))
        {
            allow_s_local_end = true;
        } else if v != cm.m && cm.sttype[vu] as i32 != EL_ST {
            jp_v = (j - jmin[vu]) as usize;
            dp_v = (d - hdmin[vu][jp_v]) as usize;
        }

        if v == cm.m || cm.sttype[vu] as i32 == E_ST || cm.sttype[vu] as i32 == EL_ST {
            match pda.pop() {
                None => break,
                Some((sj, sk, bifparent)) => {
                    j = sj;
                    d = sk;
                    let bstate = tr.state[bifparent as usize];
                    let y = cm.cnum[bstate as usize];
                    i = j - d + 1;
                    let idx = tr.add_node(i, j, y, -1, -1, bifparent);
                    tr.nxtr[bifparent as usize] = idx;
                    v = y;
                    continue;
                }
            }
        }
        if cm.sttype[vu] as i32 == B_ST {
            let k = sh.kshadow[vu][jp_v][dp_v];
            let parent = tr.n - 1;
            pda.push((j, k, parent));
            j -= k;
            d -= k;
            i = j - d + 1;
            let y = cm.cfirst[vu];
            let idx = tr.add_node(i, j, y, -1, -1, parent);
            tr.nxtl[parent as usize] = idx;
            v = y;
        } else {
            let yoffset = if allow_s_local_end { USED_EL } else { sh.yshadow[vu][jp_v][dp_v] };
            match cm.sttype[vu] as i32 {
                x if x == MP_ST => { i += 1; j -= 1; }
                x if x == ML_ST => { i += 1; }
                x if x == MR_ST => { j -= 1; }
                x if x == IL_ST => { i += 1; }
                x if x == IR_ST => { j -= 1; }
                _ => {} // D, S
            }
            d = j - i + 1;
            let parent = tr.n - 1;
            let ny = if yoffset == USED_EL {
                cm.m
            } else if yoffset == USED_LOCAL_BEGIN {
                b
            } else {
                cm.cfirst[vu] + yoffset
            };
            let idx = tr.add_node(i, j, ny, -1, -1, parent);
            tr.nxtl[parent as usize] = idx;
            v = ny;
        }
    }
    tr
}

// ============================================================================
// cm_PostCodeHB (cm_dpalign.c:5509). Returns (ppstr[0..L-1], avgpp).
// ============================================================================
fn cm_postcode_hb(cm: &CM, cp9b: &CP9Bands, l: i32, emit_mx: &HbEmitMx, tr: &Parsetree) -> (Vec<u8>, f32) {
    let imin = &cp9b.imin;
    let jmin = &cp9b.jmin;
    let mut ppstr = vec![0u8; l as usize];
    let mut sum_logp = IMPOSSIBLE;
    for x in 0..tr.n as usize {
        let v = tr.state[x] as usize;
        let i = tr.emitl[x];
        let j = tr.emitr[x];
        let stt = if v == cm.m as usize { EL_ST } else { cm.sttype[v] as i32 };
        if stt == EL_ST {
            for r in i..=j {
                let val = emit_mx.l_pp[v].as_ref().unwrap()[r as usize];
                ppstr[(r - 1) as usize] = fscore2postcode(val);
                sum_logp = flogsum(sum_logp, val);
            }
        }
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            let ip_v = (i - imin[v]) as usize;
            let val = emit_mx.l_pp[v].as_ref().unwrap()[ip_v];
            ppstr[(i - 1) as usize] = fscore2postcode(val);
            sum_logp = flogsum(sum_logp, val);
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            let jp_v = (j - jmin[v]) as usize;
            let val = emit_mx.r_pp[v].as_ref().unwrap()[jp_v];
            ppstr[(j - 1) as usize] = fscore2postcode(val);
            sum_logp = flogsum(sum_logp, val);
        }
    }
    let avgp = (sre_exp2(sum_logp) / l as f64) as f32;
    (ppstr, avgp)
}

// ============================================================================
// cm_AlignHB (cm_dpalign.c:740) driver. Non-truncated, non-sample. Bands cp9b
// must be pre-computed for this sequence. Returns (parsetree, optional PP, sc).
// ============================================================================
pub fn cm_align_hb(cm: &CM, cp9b: &CP9Bands, dsq: &[u8], l: i32, do_optacc: bool, want_pp: bool)
    -> (Parsetree, Option<Vec<u8>>, f32, f32) {
    let do_post = do_optacc || want_pp;
    let mut emit_mx: Option<HbEmitMx> = None;
    let mut ins_sc = 0.0f32;
    if do_post {
        let (ins, sc) = cm_inside_align_hb(cm, cp9b, dsq, l);
        ins_sc = sc;
        let out = cm_outside_align_hb(cm, cp9b, dsq, l, &ins);
        let post = cm_posterior_hb(cm, cp9b, l, &ins, &out);
        emit_mx = Some(cm_emitter_posterior_hb(cm, cp9b, l, &post));
    }

    let (sh, b, cyk_sc);
    if do_optacc {
        let (shx, bx, _pp) = cm_optacc_align_hb(cm, cp9b, l, emit_mx.as_ref().unwrap());
        sh = shx; b = bx; cyk_sc = ins_sc;
    } else {
        let (shx, bx, scx) = cm_cyk_inside_align_hb(cm, cp9b, dsq, l);
        sh = shx; b = bx; cyk_sc = scx;
    }
    let tr = cm_align_t_hb(cm, cp9b, l, &sh, b, do_optacc);

    // C cm_AlignHB (cm_dpalign.c:783-785): if(have_ppstr || do_optacc)
    //   cm_PostCodeHB(..., (have_ppstr) ? &ppstr : NULL, &avgpp).  have_ppstr ==
    //   want_pp for the cmalign call (ret_ppstr = do_post ? &ppstr : NULL). avgpp
    //   is returned as ret_avgpp (data->pp); ppstr only when want_pp.
    let mut avgpp = 0.0f32;
    let ppstr = if want_pp || do_optacc {
        let (pp, avg) = cm_postcode_hb(cm, cp9b, l, emit_mx.as_ref().unwrap(), &tr);
        avgpp = avg;
        if want_pp { Some(pp) } else { None }
    } else {
        None
    };
    (tr, ppstr, cyk_sc, avgpp)
}

// ============================================================================
// Parsetrees2Alignment (cm_parsetree.c:846), restricted to the cmalign call:
// do_full=TRUE, do_matchonly=FALSE, allow_trunc=FALSE, all J-mode parses.
// ============================================================================
#[inline]
fn is_gap_char(c: u8) -> bool {
    c == b'-' || c == b'.' || c == b'_'
}
/// C rightjustify (cm_parsetree.c:1798) on s[off..off+n].
fn rightjustify(s: &mut [u8], off: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut npos = n as i64 - 1;
    let mut opos = n as i64 - 1;
    while opos >= 0 {
        if is_gap_char(s[off + opos as usize]) {
            opos -= 1;
        } else {
            s[off + npos as usize] = s[off + opos as usize];
            npos -= 1;
            opos -= 1;
        }
    }
    while npos >= 0 {
        s[off + npos as usize] = b'.';
        npos -= 1;
    }
}
/// C leftjustify (cm_parsetree.c:1824) on s[off..off+n].
fn leftjustify(s: &mut [u8], off: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut npos = 0usize;
    let mut opos = 0usize;
    while opos < n {
        if is_gap_char(s[off + opos]) {
            opos += 1;
        } else {
            s[off + npos] = s[off + opos];
            npos += 1;
            opos += 1;
        }
    }
    while npos < n {
        s[off + npos] = b'.';
        npos += 1;
    }
}

/// Build the Stockholm MSA from parsetrees + PP strings. `sqdsq[i]` is the
/// sentinel-padded (1-based) digital sequence for trace i; `names[i]` its name.
/// `abc_out` is the output alphabet (RNA or DNA). Mirrors Parsetrees2Alignment
/// for the cmalign path. Returns an EslMsa ready for esl_msafile_write.
// C cm.c:1599 ModeEmitsLeft(): TRUE iff mode is TRMODE_J(3) or TRMODE_L(2).
#[inline]
fn mode_emits_left(mode: i8) -> bool {
    mode == crate::cm_trunc::TRMODE_J || mode == crate::cm_trunc::TRMODE_L
}
// C cm.c:1613 ModeEmitsRight(): TRUE iff mode is TRMODE_J(3) or TRMODE_R(1).
#[inline]
fn mode_emits_right(mode: i8) -> bool {
    mode == crate::cm_trunc::TRMODE_J || mode == crate::cm_trunc::TRMODE_R
}

pub fn parsetrees_to_alignment(
    cm: &CM,
    abc_out: &EslAlphabet,
    names: &[String],
    sqdsq: &[Vec<u8>],
    trs: &[Parsetree],
    ppstrs: &[Option<Vec<u8>>],
    do_post: bool,
    do_matchonly: bool,
    // C Parsetrees2Alignment: do_flush = (cm->align_opts & CM_ALIGN_FLUSHINSERTS)
    // leaves inserts flush L/R (skips the split); allow_trunc adds missing (~)
    // chars around truncated (is_std==false) parses (cmbuild --fins / --miss).
    do_flush: bool,
    allow_trunc: bool,
) -> EslMsa {
    let emap = create_emit_map(cm).expect("create_emit_map");
    let cmcons: CmConsensus = create_cm_consensus(cm);
    let clen = emap.clen as usize;
    let nseq = trs.len();
    let has_rf = cm.flags & crate::cm::CM_RF != 0;

    // maxil/maxel/maxir over all traces (cm_parsetree.c:941)
    let mut maxil = vec![0i32; clen + 1];
    let mut maxel = vec![0i32; clen + 1];
    let mut maxir = vec![0i32; clen + 1];

    for i in 0..nseq {
        let mut iluse = vec![0i32; clen + 1];
        let mut eluse = vec![0i32; clen + 1];
        let mut iruse = vec![0i32; clen + 1];
        let tr = &trs[i];
        let mut prvnd = 0i32;
        for tpos in 0..tr.n as usize {
            let v = tr.state[tpos] as usize;
            let mode = tr.mode[tpos];
            let stt = if v == cm.m as usize { EL_ST } else { cm.sttype[v] as i32 };
            let nd = if stt == EL_ST { prvnd } else { cm.ndidx[v] } as usize;
            match stt {
                // C cm_parsetree.c:972-973: `if(ModeEmitsLeft(mode)) iluse[...]++;`
                x if x == IL_ST => {
                    if mode_emits_left(mode) {
                        iluse[emap.lpos[nd] as usize] += 1;
                    }
                }
                // C cm_parsetree.c:979: `if(ModeEmitsRight(mode)) iruse[emap->rpos[nd]-1]++;`
                x if x == IR_ST => {
                    if mode_emits_right(mode) {
                        iruse[(emap.rpos[nd] - 1) as usize] += 1;
                    }
                }
                x if x == EL_ST => {
                    let el_len = tr.emitr[tpos] - tr.emitl[tpos] + 1;
                    eluse[emap.epos[nd] as usize] = el_len;
                }
                _ => {}
            }
            prvnd = nd as i32;
        }
        for cpos in 0..=clen {
            if iluse[cpos] > maxil[cpos] {
                maxil[cpos] = iluse[cpos];
            }
            if eluse[cpos] > maxel[cpos] {
                maxel[cpos] = eluse[cpos];
            }
            if iruse[cpos] > maxir[cpos] {
                maxir[cpos] = iruse[cpos];
            }
        }
    }

    // C: cm_parsetree.c:1021-1024 — with do_matchonly, no insert (IL/EL/IR)
    // columns are added to the alignment (alen += maxil/maxel/maxir is skipped)
    // and their emissions are not placed (the emission loop `break`s at inserts).
    // We reproduce that by zeroing the insert maxima here; the alen calc, insert
    // rejustification and the SS_cons/RF insert fills are all already guarded by
    // `max* > 0`, so they collapse to match-only columns automatically. The
    // emission loop skips the IL/EL/IR arms when do_matchonly (see below).
    if do_matchonly {
        for cpos in 0..=clen {
            maxil[cpos] = 0;
            maxel[cpos] = 0;
            maxir[cpos] = 0;
        }
    }

    // matuse (do_full=TRUE): all cpos 1..clen used; cpos 0 not.
    // maps (cm_parsetree.c:1012)
    let mut matmap = vec![-1i32; clen + 1];
    let mut ilmap = vec![0i32; clen + 1];
    let mut elmap = vec![0i32; clen + 1];
    let mut irmap = vec![0i32; clen + 1];
    let mut matuse = vec![1i32; clen + 1];
    matuse[0] = 0;
    let mut alen = 0i32;
    for cpos in 0..=clen {
        if matuse[cpos] != 0 {
            matmap[cpos] = alen;
            alen += 1;
        } else {
            matmap[cpos] = -1;
        }
        elmap[cpos] = alen;
        alen += maxel[cpos];
        ilmap[cpos] = alen;
        alen += maxil[cpos];
        alen += maxir[cpos];
        irmap[cpos] = alen - 1;
    }
    let alen = alen as usize;

    let mut msa = EslMsa::new();
    msa.nseq = nseq;
    msa.alen = alen as i64;

    let mut aseqs: Vec<Vec<u8>> = Vec::with_capacity(nseq);
    let mut ppseqs: Vec<Option<Vec<u8>>> = Vec::with_capacity(nseq);

    for i in 0..nseq {
        let tr = &trs[i];
        let dsq = &sqdsq[i];
        let do_cur_post = do_post && ppstrs[i].is_some();
        let pc = ppstrs[i].as_ref();

        let mut aseq = vec![b'.'; alen];
        let mut apc = if do_cur_post { vec![b'.'; alen] } else { Vec::new() };
        for cpos in 0..=clen {
            if matmap[cpos] != -1 {
                aseq[matmap[cpos] as usize] = b'-';
            }
        }

        let mut iluse = vec![0i32; clen + 1];
        let mut iruse = vec![0i32; clen + 1];
        let mut prvnd = 0i32;

        for tpos in 0..tr.n as usize {
            let v = tr.state[tpos] as usize;
            let mode = tr.mode[tpos];
            let stt = if v == cm.m as usize { EL_ST } else { cm.sttype[v] as i32 };
            let stid = if v == cm.m as usize { -1 } else { cm.stid[v] as i32 };
            let nd = if stt == EL_ST { prvnd } else { cm.ndidx[v] } as usize;

            match stt {
                // C cm_parsetree.c:1097-1121: MP left emitted iff ModeEmitsLeft, right
                // iff ModeEmitsRight (marginal L/R emit only one side).
                x if x == MP_ST => {
                    if mode_emits_left(mode) {
                        let cpos = emap.lpos[nd] as usize;
                        let apos = matmap[cpos] as usize;
                        let rpos = tr.emitl[tpos];
                        aseq[apos] = abc_out.sym[dsq[rpos as usize] as usize] as u8;
                        if do_cur_post {
                            apc[apos] = pc.unwrap()[(rpos - 1) as usize];
                        }
                    }
                    if mode_emits_right(mode) {
                        let cpos = emap.rpos[nd] as usize;
                        let apos = matmap[cpos] as usize;
                        let rpos = tr.emitr[tpos];
                        aseq[apos] = abc_out.sym[dsq[rpos as usize] as usize] as u8;
                        if do_cur_post {
                            apc[apos] = pc.unwrap()[(rpos - 1) as usize];
                        }
                    }
                }
                // C cm_parsetree.c:1123-1135: ML emits left iff ModeEmitsLeft.
                x if x == ML_ST => {
                    if mode_emits_left(mode) {
                        let cpos = emap.lpos[nd] as usize;
                        let apos = matmap[cpos] as usize;
                        let rpos = tr.emitl[tpos];
                        aseq[apos] = abc_out.sym[dsq[rpos as usize] as usize] as u8;
                        if do_cur_post {
                            apc[apos] = pc.unwrap()[(rpos - 1) as usize];
                        }
                    }
                }
                // C cm_parsetree.c:1137-1149: MR emits right iff ModeEmitsRight.
                x if x == MR_ST => {
                    if mode_emits_right(mode) {
                        let cpos = emap.rpos[nd] as usize;
                        let apos = matmap[cpos] as usize;
                        let rpos = tr.emitr[tpos];
                        aseq[apos] = abc_out.sym[dsq[rpos as usize] as usize] as u8;
                        if do_cur_post {
                            apc[apos] = pc.unwrap()[(rpos - 1) as usize];
                        }
                    }
                }
                // C cm_parsetree.c:1151-1164: IL emits (and iluse++) only if ModeEmitsLeft.
                x if x == IL_ST => {
                    if mode_emits_left(mode) {
                        let cpos = emap.lpos[nd] as usize;
                        let apos = (ilmap[cpos] + iluse[cpos]) as usize;
                        let rpos = tr.emitl[tpos];
                        iluse[cpos] += 1;
                        if !do_matchonly {
                            aseq[apos] =
                                (abc_out.sym[dsq[rpos as usize] as usize] as u8).to_ascii_lowercase();
                            if do_cur_post {
                                apc[apos] = pc.unwrap()[(rpos - 1) as usize];
                            }
                        }
                    }
                }
                x if x == EL_ST && !do_matchonly => {
                    let cpos = emap.epos[nd] as usize;
                    let mut apos = elmap[cpos] as usize;
                    for rpos in tr.emitl[tpos]..=tr.emitr[tpos] {
                        aseq[apos] = (abc_out.sym[dsq[rpos as usize] as usize] as u8).to_ascii_lowercase();
                        if do_cur_post {
                            apc[apos] = pc.unwrap()[(rpos - 1) as usize];
                        }
                        apos += 1;
                    }
                }
                // C cm_parsetree.c:1187-1200: IR emits (and iruse++) only if ModeEmitsRight.
                x if x == IR_ST => {
                    if mode_emits_right(mode) {
                        let cpos = (emap.rpos[nd] - 1) as usize;
                        let apos = (irmap[cpos] - iruse[cpos]) as usize;
                        let rpos = tr.emitr[tpos];
                        iruse[cpos] += 1;
                        if !do_matchonly {
                            aseq[apos] =
                                (abc_out.sym[dsq[rpos as usize] as usize] as u8).to_ascii_lowercase();
                            if do_cur_post {
                                apc[apos] = pc.unwrap()[(rpos - 1) as usize];
                            }
                        }
                    }
                }
                // C cm_parsetree.c:1202-1210: delete-column gaps, mode-guarded.
                x if x == D_ST => {
                    if (stid == MATP_D || stid == MATL_D)
                        && mode_emits_left(mode)
                        && matuse[emap.lpos[nd] as usize] != 0
                    {
                        aseq[matmap[emap.lpos[nd] as usize] as usize] = b'-';
                    }
                    if (stid == MATP_D || stid == MATR_D)
                        && mode_emits_right(mode)
                        && matuse[emap.rpos[nd] as usize] != 0
                    {
                        aseq[matmap[emap.rpos[nd] as usize] as usize] = b'-';
                    }
                }
                _ => {}
            }
            prvnd = nd as i32;
        }

        // rejustify inserts. C cm_parsetree.c:1248: with CM_ALIGN_FLUSHINSERTS
        // (--fins) the inserts are left flush (IL/EL left, IR right), skipping
        // the split entirely; default behavior splits internal inserts in half.
        if !do_flush {
        // 5' of first consensus: EL then IL flush right.
        rightjustify(&mut aseq, 0, maxel[0] as usize);
        if do_cur_post {
            rightjustify(&mut apc, 0, maxel[0] as usize);
        }
        rightjustify(&mut aseq, maxel[0] as usize, maxil[0] as usize);
        if do_cur_post {
            rightjustify(&mut apc, maxel[0] as usize, maxil[0] as usize);
        }
        // internal inserts
        for cpos in 1..clen {
            if maxel[cpos] > 1 {
                let base = (matmap[cpos] + 1) as usize;
                let mut nins = 0usize;
                let mut apos = base;
                while apos < alen && aseq[apos].is_ascii_lowercase() {
                    nins += 1;
                    apos += 1;
                }
                nins /= 2;
                let off = (matmap[cpos] + 1) as usize + nins;
                rightjustify(&mut aseq, off, maxel[cpos] as usize - nins);
                if do_cur_post {
                    rightjustify(&mut apc, off, maxel[cpos] as usize - nins);
                }
            }
            if maxil[cpos] > 1 {
                let base = (matmap[cpos] + 1 + maxel[cpos]) as usize;
                let mut nins = 0usize;
                let mut apos = base;
                while apos < alen && aseq[apos].is_ascii_lowercase() {
                    nins += 1;
                    apos += 1;
                }
                nins /= 2;
                let off = (matmap[cpos] + 1 + maxel[cpos]) as usize + nins;
                rightjustify(&mut aseq, off, maxil[cpos] as usize - nins);
                if do_cur_post {
                    rightjustify(&mut apc, off, maxil[cpos] as usize - nins);
                }
            }
            if maxir[cpos] > 1 {
                let base = (matmap[cpos + 1] - 1) as usize;
                let mut nins = 0usize;
                let mut apos = base as i64;
                while apos >= 0 && aseq[apos as usize].is_ascii_lowercase() {
                    nins += 1;
                    apos -= 1;
                }
                nins += 1;
                nins /= 2;
                let off = (matmap[cpos] + 1 + maxel[cpos] + maxil[cpos]) as usize;
                leftjustify(&mut aseq, off, maxir[cpos] as usize - nins);
                if do_cur_post {
                    leftjustify(&mut apc, off, maxir[cpos] as usize - nins);
                }
            }
        }
        // 3' of final consensus: IR flush left.
        let off = (matmap[clen] + 1 + maxel[clen] + maxil[clen]) as usize;
        leftjustify(&mut aseq, off, maxir[clen] as usize);
        if do_cur_post {
            leftjustify(&mut apc, off, maxir[clen] as usize);
        }
        } // end if !do_flush

        // C cm_parsetree.c:1306: if allow_trunc and this parse is truncated
        // (is_std==false), replace leading/trailing gap positions with the
        // missing char '~' (esl_abc_CGetMissing), per the pass's enforced ends.
        if allow_trunc && !trs[i].is_std {
            if crate::cp9::cm_pli_pass_enforces_first_res(trs[i].pass_idx) {
                for c in aseq.iter_mut() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                    *c = b'~';
                }
            }
            if crate::cp9::cm_pli_pass_enforces_final_res(trs[i].pass_idx) {
                for c in aseq.iter_mut().rev() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                    *c = b'~';
                }
            }
        }

        aseqs.push(aseq);
        ppseqs.push(if do_cur_post { Some(apc) } else { None });
    }

    // Assemble the MSA.
    for i in 0..nseq {
        msa.sqname.push(names[i].clone());
        msa.aseq.push(String::from_utf8(aseqs[i].clone()).unwrap());
        msa.wgt.push(1.0);
    }
    if do_post {
        msa.pp = Some(
            ppseqs
                .iter()
                .map(|o| o.as_ref().map(|v| String::from_utf8(v.clone()).unwrap()))
                .collect(),
        );
    }
    // author (cm_parsetree.c:1348)
    // C: snprintf(msa->au, "Infernal %s", INFERNAL_VERSION) (cm_parsetree.c:1350).
    // Provenance line; pinned to the reference C version for byte-parity.
    msa.au = Some("Infernal 1.1.5".to_string());

    // SS_cons + RF (cm_parsetree.c:1370). do_full => matuse all 1 => the ct-based
    // '.' branch is never taken; always use cmcons->cstr.
    let mut ss_cons = vec![b'.'; alen];
    let mut rf = vec![b'.'; alen];
    for cpos in 0..=clen {
        if matuse[cpos] != 0 {
            let mm = matmap[cpos] as usize;
            ss_cons[mm] = cmcons.cstr[cpos - 1];
            rf[mm] = if has_rf {
                cm.rf[cpos]
            } else {
                cmcons.cseq[cpos - 1]
            };
        }
        if maxil[cpos] > 0 {
            for apos in ilmap[cpos]..(ilmap[cpos] + maxil[cpos]) {
                ss_cons[apos as usize] = b'.';
                rf[apos as usize] = b'.';
            }
        }
        if maxel[cpos] > 0 {
            for apos in elmap[cpos]..(elmap[cpos] + maxel[cpos]) {
                ss_cons[apos as usize] = b'~';
                rf[apos as usize] = b'~';
            }
        }
        if maxir[cpos] > 0 {
            let mut apos = irmap[cpos];
            while apos > irmap[cpos] - maxir[cpos] {
                ss_cons[apos as usize] = b'.';
                rf[apos as usize] = b'.';
                apos -= 1;
            }
        }
    }
    msa.ss_cons = Some(String::from_utf8(ss_cons).unwrap());
    msa.rf = Some(String::from_utf8(rf).unwrap());

    msa
}

// ============================================================================
// cm_align_hb_ad (ADDITIVE): like cm_align_hb (do_optacc=want_pp=TRUE) but also
// returns the average posterior probability (C `adata->pp` from cm_PostCodeHB's
// ret_avgp). cmsearch's per-hit CM_ALIDISPLAY needs avgpp for the "acc" column;
// the existing cm_align_hb driver discards it. Thin wrapper over the SAME private
// posterior DP (no algorithm change). Returns (parsetree, ppstr, inside_sc, avgpp).
// ============================================================================
pub fn cm_align_hb_ad(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[u8],
    l: i32,
) -> (Parsetree, Vec<u8>, f32, f32) {
    let (ins, ins_sc) = cm_inside_align_hb(cm, cp9b, dsq, l);
    let out = cm_outside_align_hb(cm, cp9b, dsq, l, &ins);
    let post = cm_posterior_hb(cm, cp9b, l, &ins, &out);
    let emit_mx = cm_emitter_posterior_hb(cm, cp9b, l, &post);
    let (sh, b, _pp) = cm_optacc_align_hb(cm, cp9b, l, &emit_mx);
    let tr = cm_align_t_hb(cm, cp9b, l, &sh, b, true);
    let (ppstr, avg) = cm_postcode_hb(cm, cp9b, l, &emit_mx, &tr);
    (tr, ppstr, ins_sc, avg)
}
