//! cm_dpsearch_trunc — faithful port of C Infernal 1.1.5 non-banded/QDB truncated
//! DP scanners from `cm_dpsearch_trunc.c`:
//!   * `RefTrCYKScan`     (79–700)   -> [`ref_tr_cyk_scan`]     (float,   ESL_MAX)
//!   * `RefITrInsideScan` (741–1338) -> [`ref_itr_inside_scan`] (integer, ILogsum)
//!
//! Both scan a subsequence `dsq[i0..=j0]` in one of the truncated pipeline passes
//! (5P_ONLY / 3P_ONLY / 5P_AND_3P), filling the 4 marginal DP planes
//! (Jalpha/Lalpha/Ralpha + Talpha for B states, plus BEGL decks) with QDB bands,
//! applying truncation penalties (`TrPenalties`, selected by pass index / locality)
//! and the per-state marginal emission scores (`cm.lmesc/rmesc` etc, STEP 0), and
//! report greedily-resolved hits carrying the marginal mode.
//!
//! The recursion is written once over a semiring [`TrSr`] (`comb` = ESL_MAX for CYK
//! / ILogsum for Inside; `max2` = ESL_MAX for both, used in the special B/MP L,R
//! cells that C always maxes even in Inside). Faithful to the C control flow:
//!   - truncated QDB bands: dn=1 for every state, dx=min(j, dmax[v], W) (cm_mx.c:6429)
//!   - persistent ROOT row idiom: alpha planes are initialized ONCE, and the
//!     alpha[cur][0][d] cells are never reset per-j (cm.c cm_tr_scan_mx_Initialize*)
//!   - Lyoffset0/Ryoffset0 disallow IR/IL self-transits in L/R modes
//!   - greedy hit reporting + cm_hit_AllowTruncation gating (cm_tophits.c:3053).

use crate::cm::{CM, ALPHABET_SIZE_P};
use crate::constants::{
    B_ST, BEGL_S, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST, BIF_B,
    MATP_ND, MATL_ND, MATR_ND,
};
use crate::cp9::{flogsum, ilogsum, score_correction_null3, CP9Bands, INFTY};
use crate::cm_trunc::{
    TrPenalties, pty_idx_for_pass, TRMODE_J, TRMODE_L, TRMODE_R, TRMODE_T, TRMODE_UNKNOWN,
    PLI_PASS_STD_ANY,
};

const IMPOSSIBLE: f32 = -1.0e36;
const INTSCALE: f64 = 1000.0;

#[inline]
fn not_impossible_f(x: f32) -> bool {
    x > IMPOSSIBLE + 1.0
}

/// C `Scorify(sc)` (cm.c): sc==-INFTY ? IMPOSSIBLE : sc/INTSCALE.
#[inline]
fn scorify(sc: i32) -> f32 {
    if sc == -INFTY {
        IMPOSSIBLE
    } else {
        (sc as f64 / INTSCALE) as f32
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
#[inline]
fn state_left_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST || x == ML_ST || x == IL_ST => 1,
        _ => 0,
    }
}
#[inline]
fn state_is_detached(cm: &CM, v: usize) -> bool {
    (cm.stid[v + 1] as i32) == crate::constants::END_E
}

// ============================================================================
// Semiring: unifies truncated CYK (f32, ESL_MAX) and truncated Inside (i32, ILogsum).
// `comb` is the combine operator (max / logsum); `max2` is ALWAYS max (ESL_MAX),
// used for the B/MP special L,R cells that C maxes even in the Inside recursion.
// ============================================================================
trait TrSr: Copy {
    fn zero() -> Self; // IMPOSSIBLE / -INFTY
    fn base0() -> Self; // 0 (E-state base case)
    fn comb(a: Self, b: Self) -> Self; // ESL_MAX / ILogsum
    fn max2(a: Self, b: Self) -> Self; // ESL_MAX (both)
    fn addv(a: Self, b: Self) -> Self; // a + b
    fn ord(a: Self) -> f64; // raw value, for strict-> comparisons
    fn scbits(a: Self) -> f32; // reported bit score
    fn is_impossible(a: Self) -> bool;
}
impl TrSr for f32 {
    #[inline] fn zero() -> Self { IMPOSSIBLE }
    #[inline] fn base0() -> Self { 0.0 }
    #[inline] fn comb(a: Self, b: Self) -> Self { if a >= b { a } else { b } }
    #[inline] fn max2(a: Self, b: Self) -> Self { if a >= b { a } else { b } }
    #[inline] fn addv(a: Self, b: Self) -> Self { a + b }
    #[inline] fn ord(a: Self) -> f64 { a as f64 }
    #[inline] fn scbits(a: Self) -> f32 { a }
    #[inline] fn is_impossible(a: Self) -> bool { !not_impossible_f(a) }
}
impl TrSr for i32 {
    #[inline] fn zero() -> Self { -INFTY }
    #[inline] fn base0() -> Self { 0 }
    #[inline] fn comb(a: Self, b: Self) -> Self { ilogsum(a, b) }
    #[inline] fn max2(a: Self, b: Self) -> Self { if a >= b { a } else { b } }
    #[inline] fn addv(a: Self, b: Self) -> Self { a.wrapping_add(b) }
    #[inline] fn ord(a: Self) -> f64 { a as f64 }
    #[inline] fn scbits(a: Self) -> f32 { scorify(a) }
    #[inline] fn is_impossible(a: Self) -> bool { a == -INFTY }
}

/// A hit reported by a truncated scanner (window-frame coords).
#[derive(Clone, Copy, Debug)]
pub struct TrHit {
    pub i: i32,
    pub j: i32,
    pub score: f32, // null3-corrected reported bit score
    pub bias: f32,  // null3 correction applied
    pub mode: i8,   // marginal mode (TRMODE_J/L/R/T)
    pub root: i32,  // entry (truncated-begin) state
}

// QDB band indices (C SMX_*).
pub const SMX_NOQDB: usize = 0;
pub const SMX_QDB1_TIGHT: usize = 1;
pub const SMX_QDB2_LOOSE: usize = 2;

/// Per-semiring score tables handed to the generic truncated scanner.
struct TrTables<'a, S: TrSr> {
    tsc: &'a [Vec<S>],
    oesc: &'a [Vec<S>],
    lmesc: &'a [Vec<S>],
    rmesc: &'a [Vec<S>],
    endsc: Vec<S>,
    el_self: S,
    pty: Vec<S>, // truncated-begin penalty per state y (already selected)
}

fn build_float_tables<'a>(cm: &'a CM, trp: &TrPenalties, pty_idx: usize, local: bool) -> TrTables<'a, f32> {
    // C cm_dpsearch_trunc.c:544: (cm->flags & CMH_LOCAL_BEGIN) ? l_ptyAA : g_ptyAA.
    let pty = if local { &trp.l } else { &trp.g };
    TrTables {
        tsc: &cm.tsc,
        oesc: &cm.oesc,
        lmesc: &cm.lmesc,
        rmesc: &cm.rmesc,
        endsc: cm.endsc.clone(),
        el_self: cm.el_selfsc,
        pty: pty[pty_idx].clone(),
    }
}
fn build_int_tables<'a>(cm: &'a CM, trp: &TrPenalties, pty_idx: usize, local: bool) -> TrTables<'a, i32> {
    // C cm_dpsearch_trunc.c:1192: (cm->flags & CMH_LOCAL_BEGIN) ? il_ptyAA : ig_ptyAA.
    let ipty = if local { &trp.il } else { &trp.ig };
    TrTables {
        tsc: &cm.itsc,
        oesc: &cm.ioesc,
        lmesc: &cm.ilmesc,
        rmesc: &cm.irmesc,
        endsc: cm.iendsc.clone(),
        el_self: cm.iel_selfsc,
        pty: ipty[pty_idx].clone(),
    }
}

/// C `cm_TrFillFromPassIdx` (cm_dpsearch_trunc.c:3566): which of the L/R/T marginal
/// matrices must be filled for a given pass.
pub(crate) fn fill_from_pass_idx(pass_idx: i32) -> (bool, bool, bool) {
    use crate::cm_trunc::{
        PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE, PLI_PASS_5P_AND_3P_FORCE,
        PLI_PASS_5P_AND_3P_ANY,
    };
    match pass_idx {
        // PLI_PASS_5P_AND_3P_FORCE / _ANY: fill L, R, and T
        x if x == PLI_PASS_5P_AND_3P_FORCE || x == PLI_PASS_5P_AND_3P_ANY => (true, true, true),
        // 5P_ONLY: a 5' truncation — R matrix (i0 forced), no L, no T
        x if x == PLI_PASS_5P_ONLY_FORCE => (false, true, false),
        // 3P_ONLY: a 3' truncation — L matrix (j0 forced), no R, no T
        x if x == PLI_PASS_3P_ONLY_FORCE => (true, false, false),
        // standard pass: J only
        _ => (false, false, false),
    }
}

// ============================================================================
// Public entry points.
// ============================================================================

/// C `RefTrCYKScan` (cm_dpsearch_trunc.c:79). Returns (hits, vsc_root, vmode_root).
#[allow(clippy::too_many_arguments)]
pub fn ref_tr_cyk_scan(
    cm: &CM,
    trp: &TrPenalties,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    cutoff: f32,
    do_null3: bool,
    qdbidx: usize,
    pass_idx: i32,
    local: bool,
) -> (Vec<TrHit>, f32, i8, f32) {
    let pty_idx = pty_idx_for_pass(pass_idx).expect("truncated pass");
    let tab = build_float_tables(cm, trp, pty_idx, local);
    generic_tr_scan::<f32>(cm, &tab, dsq, i0, j0, cutoff, do_null3, qdbidx, pass_idx, local)
}

/// C `RefITrInsideScan` (cm_dpsearch_trunc.c:741).
#[allow(clippy::too_many_arguments)]
pub fn ref_itr_inside_scan(
    cm: &CM,
    trp: &TrPenalties,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    cutoff: f32,
    do_null3: bool,
    qdbidx: usize,
    pass_idx: i32,
    local: bool,
) -> (Vec<TrHit>, f32, i8, f32) {
    let pty_idx = pty_idx_for_pass(pass_idx).expect("truncated pass");
    let tab = build_int_tables(cm, trp, pty_idx, local);
    generic_tr_scan::<i32>(cm, &tab, dsq, i0, j0, cutoff, do_null3, qdbidx, pass_idx, local)
}

// ============================================================================
// HMM-banded truncated CYK scanner (F6 filter).
// ============================================================================

/// C `TrCYKScanHB` (cm_dpsearch_trunc.c:1387). An HMM-banded scanning TrCYK
/// implementation. Fills the 4 marginal DP planes (J/L/R/T) over the
/// HMM-band-bounded `[v][jp][dp]` layout (jp = j - jmin[v], dp = d - hdmin[v][jp]),
/// exactly as the non-truncated [`crate::cp9::fast_cyk_scan_hb`], but grafting the
/// L/R/T recursion of the byte-verified non-HB [`generic_tr_scan`] onto it. QDBs are
/// not used; bands come from `cp9b` (jmin/jmax + hdmin/hdmax) and the marginal
/// validity flags (jvalid/lvalid/rvalid/tvalid) built in Stage 1.
///
/// Returns `(hits, vsc_root, vmode_root, envi, envj)` where `vsc_root`/`vmode_root`
/// are the C `ret_sc`/`ret_mode` (best hit score + marginal mode over all 4 planes),
/// `envi`/`envj` the C envelope bounds (`-1` if none exceed `env_cutoff`), and `hits`
/// the greedily overlap-resolved reported hits.
///
/// `local` selects whether truncation is validated against local rules (mirrors
/// C's `cm->flags & CMH_LOCAL_BEGIN`); the truncation penalties themselves use the
/// GLOBAL `trp.g` arrays, matching infernox's global-only [`build_float_tables`].
#[allow(clippy::too_many_arguments)]
pub fn tr_cyk_scan_hb(
    cm: &CM,
    trp: &TrPenalties,
    cp9b: &CP9Bands,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    cutoff: f32,
    do_null3: bool,
    pass_idx: i32,
    env_cutoff: f32,
    local: bool,
) -> (Vec<TrHit>, f32, i8, i64, i64) {
    let m = cm.m as usize;
    let kp = ALPHABET_SIZE_P;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let jvalid = &cp9b.jvalid;
    let lvalid = &cp9b.lvalid;
    let rvalid = &cp9b.rvalid;
    let tvalid = &cp9b.tvalid;

    // from <pass_idx>: which marginal matrices to fill + which trunc-penalty array
    let (fill_l, fill_r, fill_t) = fill_from_pass_idx(pass_idx);
    let pty_idx = pty_idx_for_pass(pass_idx).expect("truncated pass");
    // C cm_dpsearch_trunc.c:2262/3384: (cm->flags & CMH_LOCAL_BEGIN) ? l_ptyAA : g_ptyAA.
    // `local` also selects the truncation-validity rules in allow_truncation().
    let pty: &[f32] = if local { &trp.l[pty_idx] } else { &trp.g[pty_idx] };

    // W = j0-i0+1 (C:1479). Only used to size act/bestr vectors.
    let ww = (j0 - i0 + 1) as usize;

    // precompute local-end emission scores el_scA[d] = el_selfsc * d  (C:1487)
    let mut el_sca = vec![0.0f32; ww + 1];
    for d in 0..=ww {
        el_sca[d] = cm.el_selfsc * d as f32;
    }

    // --- Build flat banded matrix layout: roff(v,jp) = flat index of (v,jp,dp=0) ---
    // identical to fast_cyk_scan_hb: one contiguous row-offset table.
    let nj = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut roff_start: Vec<usize> = vec![0; m];
    let mut roff_flat: Vec<usize> = Vec::new();
    let mut ncells: usize = 0;
    for v in 0..m {
        let njv = nj(v);
        roff_start[v] = roff_flat.len();
        for jp in 0..njv {
            roff_flat.push(ncells);
            let width = hdmax[v][jp] - hdmin[v][jp] + 1; // >=0 by -1/-2 convention
            if width > 0 {
                ncells += width as usize;
            }
        }
    }
    // row-base for (v, jp)
    let rb = |v: usize, jp: usize| -> usize { roff_flat[roff_start[v] + jp] };

    let ninf = IMPOSSIBLE;
    // The four marginal planes. C only FSets L/R/T to IMPOSSIBLE if fill_*; unfilled
    // decks are never read (every read is gated by do_{L,R,T}_*), so initializing all
    // four to IMPOSSIBLE is result-identical.
    let mut ja = vec![ninf; ncells];
    let mut la = vec![ninf; ncells];
    let mut ra = vec![ninf; ncells];
    let mut ta = vec![ninf; ncells];

    // if do_null3: pre-fill the cumulative act vector (C:1524-1535)
    let act: Vec<[f64; 4]> = if do_null3 {
        let mut act = vec![[0.0f64; 4]; ww + 1];
        for j in i0..=j0 {
            let jp = (j - i0 + 1) as usize;
            act[jp % (ww + 1)] = act[(jp - 1) % (ww + 1)];
            let dj = dsq[j as usize];
            if (dj as usize) < 4 {
                act[jp % (ww + 1)][dj as usize] += 1.0;
            }
        }
        act
    } else {
        Vec::new()
    };

    // envelope boundary variables (C:1541-1542)
    let mut envi: i64 = (j0 + 1) as i64;
    let mut envj: i64 = (i0 - 1) as i64;

    // ---- Main recursion: v = M-1 .. 1 (ROOT handled separately) (C:1545) ----
    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        let sdl = state_left_delta(stt);
        let sdr = state_right_delta(stt);
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let lmesc_v = &cm.lmesc[v];
        let rmesc_v = &cm.rmesc[v];
        let do_j_v = jvalid[v];
        let do_l_v = lvalid[v] && fill_l;
        let do_r_v = rvalid[v] && fill_r;
        let do_t_v = tvalid[v] && fill_t;

        // re-initialize J/L/R decks if we can do a local end from v (C:1561-1604)
        if not_impossible_f(cm.endsc[v]) {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let base = rb(v, jp_v);
                if do_j_v {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sd {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sd, (sd - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        ja[base + dp_v] = el_sca[(d - sd) as usize] + cm.endsc[v];
                        dp_v += 1;
                        d += 1;
                    }
                }
                if do_l_v {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdl {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sdl, (sdl - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        la[base + dp_v] = el_sca[(d - sdl) as usize] + cm.endsc[v];
                        dp_v += 1;
                        d += 1;
                    }
                }
                if do_r_v {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdr {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sdr, (sdr - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        ra[base + dp_v] = el_sca[(d - sdr) as usize] + cm.endsc[v];
                        dp_v += 1;
                        d += 1;
                    }
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let base = rb(v, jp_v);
                if do_j_v { ja[base] = 0.0; }
                if do_l_v { la[base] = 0.0; }
                if do_r_v { ra[base] = 0.0; }
            }
        } else if stt == ML_ST || stt == IL_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                // valid children: j-sdr within y's j band
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let base_v = rb(v, jp_v);
                let hdmin_vjp = hdmin[v][jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax[v][jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin_vjp) as usize;
                    // Handle J and L first (must be complete before R)
                    if do_j_v || do_l_v {
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_j_y = jvalid[y];
                            let do_l_y = lvalid[y] && fill_l;
                            if do_j_y || do_l_y {
                                let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                    let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                    let cy = rb(y, jp_y_sdr);
                                    if do_j_v && do_j_y {
                                        let cand = ja[cy + dp_y_sd] + tsc_v[yoffset];
                                        if cand > ja[base_v + dp_v] { ja[base_v + dp_v] = cand; }
                                    }
                                    if do_l_v && do_l_y {
                                        let cand = la[cy + dp_y_sd] + tsc_v[yoffset];
                                        if cand > la[base_v + dp_v] { la[base_v + dp_v] = cand; }
                                    }
                                }
                            }
                        }
                        if do_j_v {
                            ja[base_v + dp_v] += esc_v[dsq[i as usize] as usize];
                            if ja[base_v + dp_v] < ninf { ja[base_v + dp_v] = ninf; }
                        }
                        if do_l_v {
                            let e = esc_v[dsq[i as usize] as usize];
                            la[base_v + dp_v] = if d >= 2 { la[base_v + dp_v] + e } else { e };
                            if la[base_v + dp_v] < ninf { la[base_v + dp_v] = ninf; }
                        }
                    }
                    // Handle R separately (uses 'd', not 'd-sd'; disallow IL self-transit)
                    if do_r_v {
                        let mut rsc = ra[base_v + dp_v];
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_r_y = rvalid[y] && fill_r;
                            let do_j_y = jvalid[y];
                            if (do_j_y || do_r_y) && y != v {
                                let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                if d >= hdmin[y][jp_y_sdr] && d <= hdmax[y][jp_y_sdr] {
                                    let dp_y = (d - hdmin[y][jp_y_sdr]) as usize;
                                    let cy = rb(y, jp_y_sdr);
                                    if do_j_y {
                                        let cand = ja[cy + dp_y] + tsc_v[yoffset];
                                        if cand > rsc { rsc = cand; }
                                    }
                                    if do_r_y {
                                        let cand = ra[cy + dp_y] + tsc_v[yoffset];
                                        if cand > rsc { rsc = cand; }
                                    }
                                }
                            }
                        }
                        ra[base_v + dp_v] = rsc;
                    }
                    d += 1;
                }
            }
        } else if stt == MR_ST || stt == IR_ST {
            // First loop: J and R (share the same j set) (C:1715-1761)
            if do_j_v || do_r_v {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let j_sdr = j - sdr;
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                            yvalid.push(yoffset);
                        }
                    }
                    let base_v = rb(v, jp_v);
                    let hdmin_vjp = hdmin[v][jp_v];
                    let mut d = hdmin_vjp;
                    while d <= hdmax[v][jp_v] {
                        let dp_v = (d - hdmin_vjp) as usize;
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_j_y = jvalid[y];
                            let do_r_y = rvalid[y] && fill_r;
                            if do_j_y || do_r_y {
                                let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                    let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                    let cy = rb(y, jp_y_sdr);
                                    if do_j_v && do_j_y {
                                        let cand = ja[cy + dp_y_sd] + tsc_v[yoffset];
                                        if cand > ja[base_v + dp_v] { ja[base_v + dp_v] = cand; }
                                    }
                                    if do_r_v && do_r_y {
                                        let cand = ra[cy + dp_y_sd] + tsc_v[yoffset];
                                        if cand > ra[base_v + dp_v] { ra[base_v + dp_v] = cand; }
                                    }
                                }
                            }
                        }
                        if do_j_v {
                            ja[base_v + dp_v] += esc_v[dsq[j as usize] as usize];
                            if ja[base_v + dp_v] < ninf { ja[base_v + dp_v] = ninf; }
                        }
                        if do_r_v {
                            let e = esc_v[dsq[j as usize] as usize];
                            ra[base_v + dp_v] = if d >= 2 { ra[base_v + dp_v] + e } else { e };
                            if ra[base_v + dp_v] < ninf { ra[base_v + dp_v] = ninf; }
                        }
                        d += 1;
                    }
                }
            }
            // Second loop: L (uses 'j', not 'j-sdr'; disallow IR self-transit) (C:1763-1806)
            if do_l_v {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        if y != v && j >= jmin[y] && j <= jmax[y] {
                            yvalid.push(yoffset);
                        }
                    }
                    let base_v = rb(v, jp_v);
                    let hdmin_vjp = hdmin[v][jp_v];
                    let mut d = hdmin_vjp;
                    while d <= hdmax[v][jp_v] {
                        let dp_v = (d - hdmin_vjp) as usize;
                        let mut lsc = la[base_v + dp_v];
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_l_y = lvalid[y] && fill_l;
                            let do_j_y = jvalid[y];
                            if do_l_y || do_j_y {
                                let jp_y = (j - jmin[y]) as usize;
                                if d >= hdmin[y][jp_y] && d <= hdmax[y][jp_y] {
                                    let dp_y = (d - hdmin[y][jp_y]) as usize;
                                    let cy = rb(y, jp_y);
                                    if do_j_y {
                                        let cand = ja[cy + dp_y] + tsc_v[yoffset];
                                        if cand > lsc { lsc = cand; }
                                    }
                                    if do_l_y {
                                        let cand = la[cy + dp_y] + tsc_v[yoffset];
                                        if cand > lsc { lsc = cand; }
                                    }
                                }
                            }
                        }
                        la[base_v + dp_v] = lsc;
                        d += 1;
                    }
                }
            }
        } else if stt == MP_ST {
            // for y { J&R loop; L loop } (C:1815-1932)
            for yoffset in 0..cnum {
                let y = cfirst + yoffset;
                let do_j_y = jvalid[y];
                let do_l_y = lvalid[y] && fill_l;
                let do_r_y = rvalid[y] && fill_r;
                let tsc = tsc_v[yoffset];

                if (do_j_v && do_j_y) || (do_r_v && (do_j_y || do_r_y)) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = jn - jmin[v];
                    let mut jp_y_sdr = jn - jmin[y] - sdr;
                    while jp_v <= jx - jmin[v] {
                        let jpv = jp_v as usize;
                        let jpysdr = jp_y_sdr as usize;
                        if do_j_v && do_j_y {
                            let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                            let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                            let mut dp_v = dn - hdmin[v][jpv];
                            let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                            let bv = rb(v, jpv);
                            let by = rb(y, jpysdr);
                            while dp_v <= dx - hdmin[v][jpv] {
                                let cand = ja[by + dp_y_sd as usize] + tsc;
                                if cand > ja[bv + dp_v as usize] { ja[bv + dp_v as usize] = cand; }
                                dp_v += 1;
                                dp_y_sd += 1;
                            }
                        }
                        if do_r_v && (do_r_y || do_j_y) {
                            let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sdr);
                            let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sdr);
                            let mut dp_v = dn - hdmin[v][jpv];
                            let mut dp_y_sdr = dn - hdmin[y][jpysdr] - sdr;
                            let bv = rb(v, jpv);
                            let by = rb(y, jpysdr);
                            while dp_v <= dx - hdmin[v][jpv] {
                                if do_j_y {
                                    let cand = ja[by + dp_y_sdr as usize] + tsc;
                                    if cand > ra[bv + dp_v as usize] { ra[bv + dp_v as usize] = cand; }
                                }
                                if do_r_y {
                                    let cand = ra[by + dp_y_sdr as usize] + tsc;
                                    if cand > ra[bv + dp_v as usize] { ra[bv + dp_v as usize] = cand; }
                                }
                                dp_v += 1;
                                dp_y_sdr += 1;
                            }
                        }
                        jp_v += 1;
                        jp_y_sdr += 1;
                    }
                }

                if do_l_v && (do_l_y || do_j_y) {
                    let jn = jmin[v].max(jmin[y]);
                    let jx = jmax[v].min(jmax[y]);
                    let mut jp_v = jn - jmin[v];
                    let mut jp_y = jn - jmin[y];
                    while jp_v <= jx - jmin[v] {
                        let jpv = jp_v as usize;
                        let jpy = jp_y as usize;
                        let dn = hdmin[v][jpv].max(hdmin[y][jpy] + sdr);
                        let dx = hdmax[v][jpv].min(hdmax[y][jpy] + sdr);
                        let mut dp_v = dn - hdmin[v][jpv];
                        let mut dp_y_sdr = dn - hdmin[y][jpy] - sdr;
                        let bv = rb(v, jpv);
                        let by = rb(y, jpy);
                        while dp_v <= dx - hdmin[v][jpv] {
                            if do_j_y {
                                let cand = ja[by + dp_y_sdr as usize] + tsc;
                                if cand > la[bv + dp_v as usize] { la[bv + dp_v as usize] = cand; }
                            }
                            if do_l_y {
                                let cand = la[by + dp_y_sdr as usize] + tsc;
                                if cand > la[bv + dp_v as usize] { la[bv + dp_v as usize] = cand; }
                            }
                            dp_v += 1;
                            dp_y_sdr += 1;
                        }
                        jp_v += 1;
                        jp_y += 1;
                    }
                }
            }
            // add in emission scores (C:1934-1957)
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut i = j - hdmin[v][jp_v] + 1;
                let base_v = rb(v, jp_v);
                let mut dp_v = 0usize;
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    if d >= 2 {
                        if do_j_v {
                            let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                            ja[base_v + dp_v] += esc_v[idx];
                        }
                        if do_l_v { la[base_v + dp_v] += lmesc_v[dsq[i as usize] as usize]; }
                        if do_r_v { ra[base_v + dp_v] += rmesc_v[dsq[j as usize] as usize]; }
                    } else {
                        if do_j_v { ja[base_v + dp_v] = ninf; }
                        if do_l_v { la[base_v + dp_v] = lmesc_v[dsq[i as usize] as usize]; }
                        if do_r_v { ra[base_v + dp_v] = rmesc_v[dsq[j as usize] as usize]; }
                    }
                    i -= 1;
                    dp_v += 1;
                    d += 1;
                }
            }
            // ensure all cells >= IMPOSSIBLE (C:1958-1966)
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let base_v = rb(v, jp_v);
                let width = (hdmax[v][jp_v] - hdmin[v][jp_v]).max(-1);
                for dp_v in 0..=width {
                    if width < 0 { break; }
                    let c = base_v + dp_v as usize;
                    if do_j_v && ja[c] < ninf { ja[c] = ninf; }
                    if do_l_v && la[c] < ninf { la[c] = ninf; }
                    if do_r_v && ra[c] < ninf { ra[c] = ninf; }
                }
            }
        } else if stt != B_ST {
            // D, S states (no self-transit, no emission) (C:1968-2030)
            for yoffset in 0..cnum {
                let y = cfirst + yoffset;
                let do_j_y = jvalid[y];
                let do_l_y = lvalid[y] && fill_l;
                let do_r_y = rvalid[y] && fill_r;
                let tsc = tsc_v[yoffset];
                if (do_j_v && do_j_y) || (do_l_v && do_l_y) || (do_r_v && do_r_y) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = jn - jmin[v];
                    let mut jp_y_sdr = jn - jmin[y] - sdr;
                    while jp_v <= jx - jmin[v] {
                        let jpv = jp_v as usize;
                        let jpysdr = jp_y_sdr as usize;
                        let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                        let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                        let dpn = dn - hdmin[v][jpv];
                        let mut dp_v = dpn;
                        let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                        let bv = rb(v, jpv);
                        let by = rb(y, jpysdr);
                        while dp_v <= dx - hdmin[v][jpv] {
                            if do_j_v && do_j_y {
                                let cand = ja[by + dp_y_sd as usize] + tsc;
                                if cand > ja[bv + dp_v as usize] { ja[bv + dp_v as usize] = cand; }
                            }
                            if do_l_v && do_l_y {
                                let cand = la[by + dp_y_sd as usize] + tsc;
                                if cand > la[bv + dp_v as usize] { la[bv + dp_v as usize] = cand; }
                            }
                            if do_r_v && do_r_y {
                                let cand = ra[by + dp_y_sd as usize] + tsc;
                                if cand > ra[bv + dp_v as usize] { ra[bv + dp_v as usize] = cand; }
                            }
                            // d == 0: force L and R to IMPOSSIBLE (C:2020-2024)
                            if dp_v == dpn && dn == 0 {
                                if do_l_v { la[bv + dp_v as usize] = ninf; }
                                if do_r_v { ra[bv + dp_v as usize] = ninf; }
                            }
                            dp_v += 1;
                            dp_y_sd += 1;
                        }
                        jp_v += 1;
                        jp_y_sdr += 1;
                    }
                }
            }
        } else {
            // B_st (C:2031-2156)
            let y = cfirst; // left subtree (BEGL_S)
            let z = cnum; // right subtree (BEGR_S)
            let do_j_y = jvalid[y];
            let do_l_y = lvalid[y] && fill_l;
            let do_r_y = rvalid[y] && fill_r;
            let do_j_z = jvalid[z];
            let do_l_z = lvalid[z] && fill_l;
            let do_r_z = rvalid[z] && fill_r;

            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let hdmin_zjp = hdmin[z][jp_z];
                let kn = (j - jmax[y]).max(hdmin_zjp).max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                let base_v = rb(v, jp_v);
                let base_z = rb(z, jp_z);
                let hdmin_vjp = hdmin[v][jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax[v][jp_v] {
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let jp_y_mk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jp_y_mk] && k <= d - hdmin[y][jp_y_mk] {
                            let kp_z = (k - hdmin_zjp) as usize;
                            let dp_y = d - hdmin[y][jp_y_mk];
                            let cy = rb(y, jp_y_mk) + (dp_y - k) as usize;
                            let cz = base_z + kp_z;
                            if do_j_v && do_j_y && do_j_z {
                                let cand = ja[cy] + ja[cz];
                                if cand > ja[base_v + dp_v] { ja[base_v + dp_v] = cand; }
                            }
                            if do_l_v && do_j_y && do_l_z {
                                let cand = ja[cy] + la[cz];
                                if cand > la[base_v + dp_v] { la[base_v + dp_v] = cand; }
                            }
                            if do_r_v && do_r_y && do_j_z {
                                let cand = ra[cy] + ja[cz];
                                if cand > ra[base_v + dp_v] { ra[base_v + dp_v] = cand; }
                            }
                            if k != 0 && k != d && do_t_v && do_r_y && do_l_z {
                                let cand = ra[cy] + la[cz];
                                if cand > ta[base_v + dp_v] { ta[base_v + dp_v] = cand; }
                            }
                        }
                        k += 1;
                    }
                    d += 1;
                }
            }
            // special case L: Lalpha[v] from y (J or L), independent of z (C:2117-2136)
            if do_l_v && (do_j_y || do_l_y) {
                let jn = jmin[v].max(jmin[y]);
                let jx = jmax[v].min(jmax[y]);
                for j in jn..=jx {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_y = (j - jmin[y]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    let base_v = rb(v, jp_v);
                    let base_y = rb(y, jp_y);
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        if do_j_y {
                            let cand = ja[base_y + dp_y];
                            if cand > la[base_v + dp_v] { la[base_v + dp_v] = cand; }
                        }
                        if do_l_y {
                            let cand = la[base_y + dp_y];
                            if cand > la[base_v + dp_v] { la[base_v + dp_v] = cand; }
                        }
                        d += 1;
                    }
                }
            }
            // special case R: Ralpha[v] from z (J or R), independent of y (C:2137-2156)
            if do_r_v && (do_j_z || do_r_z) {
                let jn = jmin[v].max(jmin[z]);
                let jx = jmax[v].min(jmax[z]);
                for j in jn..=jx {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_z = (j - jmin[z]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[z][jp_z]);
                    let dx = hdmax[v][jp_v].min(hdmax[z][jp_z]);
                    let base_v = rb(v, jp_v);
                    let base_z = rb(z, jp_z);
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_z = (d - hdmin[z][jp_z]) as usize;
                        if do_j_z {
                            let cand = ja[base_z + dp_z];
                            if cand > ra[base_v + dp_v] { ra[base_v + dp_v] = cand; }
                        }
                        if do_r_z {
                            let cand = ra[base_z + dp_z];
                            if cand > ra[base_v + dp_v] { ra[base_v + dp_v] = cand; }
                        }
                        d += 1;
                    }
                }
            }
        }
    }

    // ---- ROOT_S (v=0): truncated begins + hit reporting (C:2217-2328) ----
    let do_j_0 = jvalid[0];
    let do_l_0 = lvalid[0] && fill_l;
    let do_r_0 = rvalid[0] && fill_r;
    let do_t_0 = tvalid[0] && fill_t;

    let mut tmp_hits: Vec<TrHit> = Vec::new();
    let mut bestr = vec![0i32; ww + 1];
    let mut bestsc = vec![IMPOSSIBLE; ww + 1];
    let mut bestmode = vec![TRMODE_UNKNOWN; ww + 1];

    for j in jmin[0]..=jmax[0] {
        let jp_v = (j - jmin[0]) as usize;
        let base_v = rb(0, jp_v);
        for d in 0..=ww {
            bestr[d] = 0;
            bestsc[d] = IMPOSSIBLE;
            bestmode[d] = TRMODE_UNKNOWN;
        }
        for y in 1..m {
            let trpenalty = pty[y];
            if not_impossible_f(trpenalty) && j >= jmin[y] && j <= jmax[y] {
                let do_j_y = jvalid[y];
                let do_l_y = lvalid[y] && fill_l;
                let do_r_y = rvalid[y] && fill_r;
                let do_t_y = tvalid[y] && fill_t;
                let jp_y = (j - jmin[y]) as usize;
                let dn = hdmin[0][jp_v].max(hdmin[y][jp_y]);
                let dx = hdmax[0][jp_v].min(hdmax[y][jp_y]);
                let cy = rb(y, jp_y);
                if do_j_0 && do_j_y {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = ja[cy + dp_y] + trpenalty;
                        if sc > ja[base_v + dp_v] {
                            ja[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_J;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
                if do_l_0 && do_l_y {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = la[cy + dp_y] + trpenalty;
                        if sc > la[base_v + dp_v] {
                            la[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_L;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
                if do_r_0 && do_r_y {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = ra[cy + dp_y] + trpenalty;
                        if sc > ra[base_v + dp_v] {
                            ra[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_R;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
                if do_t_0 && do_t_y && cm.sttype[y] as i32 == B_ST {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = ta[cy + dp_y] + trpenalty;
                        if sc > ta[base_v + dp_v] {
                            ta[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_T;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
            }
        }
        // report hits greedily to tmp_hits (C ReportHitsGreedily branch)
        report_hits_greedily(
            cm, pass_idx, local, j, hdmin[0][jp_v], hdmax[0][jp_v], &bestsc, &bestr, &bestmode,
            ww, if do_null3 { Some(&act) } else { None }, i0, j0, cutoff, &mut tmp_hits,
        );
    }

    // ---- find best scoring hit + envelope boundaries (C:2340-2382) ----
    let mut vsc_root = IMPOSSIBLE;
    let mut vmode_root = TRMODE_UNKNOWN;
    let jpx = jmax[0] - jmin[0];
    for jp_v_i in 0..=jpx.max(-1) {
        if jmax[0] < jmin[0] { break; }
        let jp_v = jp_v_i as usize;
        let j = jp_v as i32 + jmin[0];
        let base_v = rb(0, jp_v);
        let dpx = hdmax[0][jp_v] - hdmin[0][jp_v];
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let c = base_v + dp_v as usize;
            if do_j_0 && ja[c] > vsc_root { vsc_root = ja[c]; vmode_root = TRMODE_J; }
            if do_l_0 && la[c] > vsc_root { vsc_root = la[c]; vmode_root = TRMODE_L; }
            if do_r_0 && ra[c] > vsc_root { vsc_root = ra[c]; vmode_root = TRMODE_R; }
            if do_t_0 && ta[c] > vsc_root { vsc_root = ta[c]; vmode_root = TRMODE_T; }
        }
        // envelope
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let c = base_v + dp_v as usize;
            if (do_j_0 && ja[c] >= env_cutoff)
                || (do_l_0 && la[c] >= env_cutoff)
                || (do_r_0 && ra[c] >= env_cutoff)
                || (do_t_0 && ta[c] >= env_cutoff)
            {
                let i = j - (dp_v + hdmin[0][jp_v]) + 1;
                if (i as i64) < envi { envi = i as i64; }
                if (j as i64) > envj { envj = j as i64; }
            }
        }
    }

    let ret_envi = if envi == (j0 + 1) as i64 { -1 } else { envi };
    let ret_envj = if envj == (i0 - 1) as i64 { -1 } else { envj };

    // greedy overlap removal (C: SortForOverlapRemoval + RemoveOrMarkOverlaps)
    let kept = remove_overlaps_greedy_tr(tmp_hits);
    (kept, vsc_root, vmode_root, ret_envi, ret_envj)
}

// ============================================================================
// HMM-banded truncated Inside scanner (F7 final stage).
// ============================================================================

/// C `FTrInsideScanHB` (cm_dpsearch_trunc.c:2507). The log-sum sibling of
/// [`tr_cyk_scan_hb`]: byte-for-byte the SAME band layout, init, ROOT-begin and
/// final `vsc_root` selection, EXCEPT every DP child-combine step accumulates with
/// `FLogsum` (cp9::flogsum) instead of `ESL_MAX`.
///
/// VERIFIED against C which of the two operators each site uses:
///   * Child-combine over transitions y (ML/IL J,L,R; MR/IR J,R,L; MP J,R,L; D/S
///     J,L,R), over B split-points k (J,L,R,T) and the two B child-only inherit
///     loops (special L from y, special R from z): **FLogsum**
///     (C:2778-2779,2810-2811,2866-2867,2914-2915,2980,3003-3004,3047-3048,
///      3136-3138,3221-3225,3252-3253,3272-3273).
///   * The `ESL_MAX(x, IMPOSSIBLE)` clamps after ML/IL, MR/IR emission adds and the
///     MP "ensure >= IMPOSSIBLE" pass (C:2785,2789,2873,2877,3082-3084) and the
///     emission `+=`/assignments themselves: **unchanged** (identical to CYK).
///   * The ROOT_S (v=0) truncated-begin `if (sc > {J,L,R,T}alpha[0]...)` update
///     (C:3383,3398,3413,3428) and the final best-over-4-planes `> vsc_root`
///     selection (C:3470-3485): **still MAX** — confirming the Stage-2 speculation.
///
/// Returns `(hits, vsc_root, vmode_root, envi, envj)` exactly like [`tr_cyk_scan_hb`].
#[allow(clippy::too_many_arguments)]
pub fn ftr_inside_scan_hb(
    cm: &CM,
    trp: &TrPenalties,
    cp9b: &CP9Bands,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    cutoff: f32,
    do_null3: bool,
    pass_idx: i32,
    env_cutoff: f32,
    local: bool,
) -> (Vec<TrHit>, f32, i8, i64, i64) {
    let m = cm.m as usize;
    let kp = ALPHABET_SIZE_P;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let jvalid = &cp9b.jvalid;
    let lvalid = &cp9b.lvalid;
    let rvalid = &cp9b.rvalid;
    let tvalid = &cp9b.tvalid;

    let (fill_l, fill_r, fill_t) = fill_from_pass_idx(pass_idx);
    let pty_idx = pty_idx_for_pass(pass_idx).expect("truncated pass");
    // C cm_dpsearch_trunc.c:3384: (cm->flags & CMH_LOCAL_BEGIN) ? l_ptyAA : g_ptyAA.
    // `local` also selects the truncation-validity rules in allow_truncation().
    let pty: &[f32] = if local { &trp.l[pty_idx] } else { &trp.g[pty_idx] };

    let ww = (j0 - i0 + 1) as usize;

    // precompute local-end emission scores el_scA[d] = el_selfsc * d (C:2607)
    let mut el_sca = vec![0.0f32; ww + 1];
    for d in 0..=ww {
        el_sca[d] = cm.el_selfsc * d as f32;
    }

    // --- Build flat banded matrix layout: roff(v,jp) = flat index of (v,jp,dp=0) ---
    let nj = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut roff_start: Vec<usize> = vec![0; m];
    let mut roff_flat: Vec<usize> = Vec::new();
    let mut ncells: usize = 0;
    for v in 0..m {
        let njv = nj(v);
        roff_start[v] = roff_flat.len();
        for jp in 0..njv {
            roff_flat.push(ncells);
            let width = hdmax[v][jp] - hdmin[v][jp] + 1;
            if width > 0 {
                ncells += width as usize;
            }
        }
    }
    let rb = |v: usize, jp: usize| -> usize { roff_flat[roff_start[v] + jp] };

    let ninf = IMPOSSIBLE;
    // Four marginal planes, all IMPOSSIBLE-init (C:2620-2623 FSets J always, L/R/T
    // only if fill_*; unfilled decks are never read since reads are do_{L,R,T}_-gated).
    let mut ja = vec![ninf; ncells];
    let mut la = vec![ninf; ncells];
    let mut ra = vec![ninf; ncells];
    let mut ta = vec![ninf; ncells];

    // if do_null3: pre-fill the cumulative act vector (C:2643-2656)
    let act: Vec<[f64; 4]> = if do_null3 {
        let mut act = vec![[0.0f64; 4]; ww + 1];
        for j in i0..=j0 {
            let jp = (j - i0 + 1) as usize;
            act[jp % (ww + 1)] = act[(jp - 1) % (ww + 1)];
            let dj = dsq[j as usize];
            if (dj as usize) < 4 {
                act[jp % (ww + 1)][dj as usize] += 1.0;
            }
        }
        act
    } else {
        Vec::new()
    };

    // envelope boundary variables (C:2661-2662)
    let mut envi: i64 = (j0 + 1) as i64;
    let mut envj: i64 = (i0 - 1) as i64;

    // ---- Main recursion: v = M-1 .. 1 (ROOT handled separately) (C:2665) ----
    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        let sdl = state_left_delta(stt);
        let sdr = state_right_delta(stt);
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let lmesc_v = &cm.lmesc[v];
        let rmesc_v = &cm.rmesc[v];
        let do_j_v = jvalid[v];
        let do_l_v = lvalid[v] && fill_l;
        let do_r_v = rvalid[v] && fill_r;
        let do_t_v = tvalid[v] && fill_t;

        // re-initialize J/L/R decks if we can do a local end from v (C:2681-2724)
        if not_impossible_f(cm.endsc[v]) {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let base = rb(v, jp_v);
                if do_j_v {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sd {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sd, (sd - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        ja[base + dp_v] = el_sca[(d - sd) as usize] + cm.endsc[v];
                        dp_v += 1;
                        d += 1;
                    }
                }
                if do_l_v {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdl {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sdl, (sdl - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        la[base + dp_v] = el_sca[(d - sdl) as usize] + cm.endsc[v];
                        dp_v += 1;
                        d += 1;
                    }
                }
                if do_r_v {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdr {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sdr, (sdr - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        ra[base + dp_v] = el_sca[(d - sdr) as usize] + cm.endsc[v];
                        dp_v += 1;
                        d += 1;
                    }
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let base = rb(v, jp_v);
                if do_j_v { ja[base] = 0.0; }
                if do_l_v { la[base] = 0.0; }
                if do_r_v { ra[base] = 0.0; }
            }
        } else if stt == ML_ST || stt == IL_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let base_v = rb(v, jp_v);
                let hdmin_vjp = hdmin[v][jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax[v][jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin_vjp) as usize;
                    // Handle J and L first (must be complete before R) — FLogsum (C:2778-2779)
                    if do_j_v || do_l_v {
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_j_y = jvalid[y];
                            let do_l_y = lvalid[y] && fill_l;
                            if do_j_y || do_l_y {
                                let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                    let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                    let cy = rb(y, jp_y_sdr);
                                    if do_j_v && do_j_y {
                                        ja[base_v + dp_v] = flogsum(ja[base_v + dp_v], ja[cy + dp_y_sd] + tsc_v[yoffset]);
                                    }
                                    if do_l_v && do_l_y {
                                        la[base_v + dp_v] = flogsum(la[base_v + dp_v], la[cy + dp_y_sd] + tsc_v[yoffset]);
                                    }
                                }
                            }
                        }
                        if do_j_v {
                            ja[base_v + dp_v] += esc_v[dsq[i as usize] as usize];
                            if ja[base_v + dp_v] < ninf { ja[base_v + dp_v] = ninf; }
                        }
                        if do_l_v {
                            let e = esc_v[dsq[i as usize] as usize];
                            la[base_v + dp_v] = if d >= 2 { la[base_v + dp_v] + e } else { e };
                            if la[base_v + dp_v] < ninf { la[base_v + dp_v] = ninf; }
                        }
                    }
                    // Handle R separately (uses 'd'; disallow IL self-transit) — FLogsum (C:2810-2811)
                    if do_r_v {
                        let mut rsc = ra[base_v + dp_v];
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_r_y = rvalid[y] && fill_r;
                            let do_j_y = jvalid[y];
                            if (do_j_y || do_r_y) && y != v {
                                let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                if d >= hdmin[y][jp_y_sdr] && d <= hdmax[y][jp_y_sdr] {
                                    let dp_y = (d - hdmin[y][jp_y_sdr]) as usize;
                                    let cy = rb(y, jp_y_sdr);
                                    if do_j_y {
                                        rsc = flogsum(rsc, ja[cy + dp_y] + tsc_v[yoffset]);
                                    }
                                    if do_r_y {
                                        rsc = flogsum(rsc, ra[cy + dp_y] + tsc_v[yoffset]);
                                    }
                                }
                            }
                        }
                        ra[base_v + dp_v] = rsc;
                    }
                    d += 1;
                }
            }
        } else if stt == MR_ST || stt == IR_ST {
            // First loop: J and R (share the same j set) — FLogsum (C:2866-2867)
            if do_j_v || do_r_v {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let j_sdr = j - sdr;
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                            yvalid.push(yoffset);
                        }
                    }
                    let base_v = rb(v, jp_v);
                    let hdmin_vjp = hdmin[v][jp_v];
                    let mut d = hdmin_vjp;
                    while d <= hdmax[v][jp_v] {
                        let dp_v = (d - hdmin_vjp) as usize;
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_j_y = jvalid[y];
                            let do_r_y = rvalid[y] && fill_r;
                            if do_j_y || do_r_y {
                                let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                    let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                    let cy = rb(y, jp_y_sdr);
                                    if do_j_v && do_j_y {
                                        ja[base_v + dp_v] = flogsum(ja[base_v + dp_v], ja[cy + dp_y_sd] + tsc_v[yoffset]);
                                    }
                                    if do_r_v && do_r_y {
                                        ra[base_v + dp_v] = flogsum(ra[base_v + dp_v], ra[cy + dp_y_sd] + tsc_v[yoffset]);
                                    }
                                }
                            }
                        }
                        if do_j_v {
                            ja[base_v + dp_v] += esc_v[dsq[j as usize] as usize];
                            if ja[base_v + dp_v] < ninf { ja[base_v + dp_v] = ninf; }
                        }
                        if do_r_v {
                            let e = esc_v[dsq[j as usize] as usize];
                            ra[base_v + dp_v] = if d >= 2 { ra[base_v + dp_v] + e } else { e };
                            if ra[base_v + dp_v] < ninf { ra[base_v + dp_v] = ninf; }
                        }
                        d += 1;
                    }
                }
            }
            // Second loop: L (uses 'j'; disallow IR self-transit) — FLogsum (C:2914-2915)
            if do_l_v {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        if y != v && j >= jmin[y] && j <= jmax[y] {
                            yvalid.push(yoffset);
                        }
                    }
                    let base_v = rb(v, jp_v);
                    let hdmin_vjp = hdmin[v][jp_v];
                    let mut d = hdmin_vjp;
                    while d <= hdmax[v][jp_v] {
                        let dp_v = (d - hdmin_vjp) as usize;
                        let mut lsc = la[base_v + dp_v];
                        for &yoffset in &yvalid {
                            let y = cfirst + yoffset;
                            let do_l_y = lvalid[y] && fill_l;
                            let do_j_y = jvalid[y];
                            if do_l_y || do_j_y {
                                let jp_y = (j - jmin[y]) as usize;
                                if d >= hdmin[y][jp_y] && d <= hdmax[y][jp_y] {
                                    let dp_y = (d - hdmin[y][jp_y]) as usize;
                                    let cy = rb(y, jp_y);
                                    if do_j_y {
                                        lsc = flogsum(lsc, ja[cy + dp_y] + tsc_v[yoffset]);
                                    }
                                    if do_l_y {
                                        lsc = flogsum(lsc, la[cy + dp_y] + tsc_v[yoffset]);
                                    }
                                }
                            }
                        }
                        la[base_v + dp_v] = lsc;
                        d += 1;
                    }
                }
            }
        } else if stt == MP_ST {
            // for y { J&R loop; L loop } — FLogsum (C:2980,3003-3004,3047-3048)
            for yoffset in 0..cnum {
                let y = cfirst + yoffset;
                let do_j_y = jvalid[y];
                let do_l_y = lvalid[y] && fill_l;
                let do_r_y = rvalid[y] && fill_r;
                let tsc = tsc_v[yoffset];

                if (do_j_v && do_j_y) || (do_r_v && (do_j_y || do_r_y)) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = jn - jmin[v];
                    let mut jp_y_sdr = jn - jmin[y] - sdr;
                    while jp_v <= jx - jmin[v] {
                        let jpv = jp_v as usize;
                        let jpysdr = jp_y_sdr as usize;
                        if do_j_v && do_j_y {
                            let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                            let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                            let mut dp_v = dn - hdmin[v][jpv];
                            let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                            let bv = rb(v, jpv);
                            let by = rb(y, jpysdr);
                            while dp_v <= dx - hdmin[v][jpv] {
                                ja[bv + dp_v as usize] = flogsum(ja[bv + dp_v as usize], ja[by + dp_y_sd as usize] + tsc);
                                dp_v += 1;
                                dp_y_sd += 1;
                            }
                        }
                        if do_r_v && (do_r_y || do_j_y) {
                            let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sdr);
                            let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sdr);
                            let mut dp_v = dn - hdmin[v][jpv];
                            let mut dp_y_sdr = dn - hdmin[y][jpysdr] - sdr;
                            let bv = rb(v, jpv);
                            let by = rb(y, jpysdr);
                            while dp_v <= dx - hdmin[v][jpv] {
                                if do_j_y {
                                    ra[bv + dp_v as usize] = flogsum(ra[bv + dp_v as usize], ja[by + dp_y_sdr as usize] + tsc);
                                }
                                if do_r_y {
                                    ra[bv + dp_v as usize] = flogsum(ra[bv + dp_v as usize], ra[by + dp_y_sdr as usize] + tsc);
                                }
                                dp_v += 1;
                                dp_y_sdr += 1;
                            }
                        }
                        jp_v += 1;
                        jp_y_sdr += 1;
                    }
                }

                if do_l_v && (do_l_y || do_j_y) {
                    let jn = jmin[v].max(jmin[y]);
                    let jx = jmax[v].min(jmax[y]);
                    let mut jp_v = jn - jmin[v];
                    let mut jp_y = jn - jmin[y];
                    while jp_v <= jx - jmin[v] {
                        let jpv = jp_v as usize;
                        let jpy = jp_y as usize;
                        let dn = hdmin[v][jpv].max(hdmin[y][jpy] + sdr);
                        let dx = hdmax[v][jpv].min(hdmax[y][jpy] + sdr);
                        let mut dp_v = dn - hdmin[v][jpv];
                        let mut dp_y_sdr = dn - hdmin[y][jpy] - sdr;
                        let bv = rb(v, jpv);
                        let by = rb(y, jpy);
                        while dp_v <= dx - hdmin[v][jpv] {
                            if do_j_y {
                                la[bv + dp_v as usize] = flogsum(la[bv + dp_v as usize], ja[by + dp_y_sdr as usize] + tsc);
                            }
                            if do_l_y {
                                la[bv + dp_v as usize] = flogsum(la[bv + dp_v as usize], la[by + dp_y_sdr as usize] + tsc);
                            }
                            dp_v += 1;
                            dp_y_sdr += 1;
                        }
                        jp_v += 1;
                        jp_y += 1;
                    }
                }
            }
            // add in emission scores (C:3054-3077) — unchanged (identical to CYK)
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut i = j - hdmin[v][jp_v] + 1;
                let base_v = rb(v, jp_v);
                let mut dp_v = 0usize;
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    if d >= 2 {
                        if do_j_v {
                            let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                            ja[base_v + dp_v] += esc_v[idx];
                        }
                        if do_l_v { la[base_v + dp_v] += lmesc_v[dsq[i as usize] as usize]; }
                        if do_r_v { ra[base_v + dp_v] += rmesc_v[dsq[j as usize] as usize]; }
                    } else {
                        if do_j_v { ja[base_v + dp_v] = ninf; }
                        if do_l_v { la[base_v + dp_v] = lmesc_v[dsq[i as usize] as usize]; }
                        if do_r_v { ra[base_v + dp_v] = rmesc_v[dsq[j as usize] as usize]; }
                    }
                    i -= 1;
                    dp_v += 1;
                    d += 1;
                }
            }
            // ensure all cells >= IMPOSSIBLE (C:3079-3086) — MAX clamp, unchanged
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let base_v = rb(v, jp_v);
                let width = (hdmax[v][jp_v] - hdmin[v][jp_v]).max(-1);
                for dp_v in 0..=width {
                    if width < 0 { break; }
                    let c = base_v + dp_v as usize;
                    if do_j_v && ja[c] < ninf { ja[c] = ninf; }
                    if do_l_v && la[c] < ninf { la[c] = ninf; }
                    if do_r_v && ra[c] < ninf { ra[c] = ninf; }
                }
            }
        } else if stt != B_ST {
            // D, S states (no self-transit, no emission) — FLogsum (C:3136-3138)
            for yoffset in 0..cnum {
                let y = cfirst + yoffset;
                let do_j_y = jvalid[y];
                let do_l_y = lvalid[y] && fill_l;
                let do_r_y = rvalid[y] && fill_r;
                let tsc = tsc_v[yoffset];
                if (do_j_v && do_j_y) || (do_l_v && do_l_y) || (do_r_v && do_r_y) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = jn - jmin[v];
                    let mut jp_y_sdr = jn - jmin[y] - sdr;
                    while jp_v <= jx - jmin[v] {
                        let jpv = jp_v as usize;
                        let jpysdr = jp_y_sdr as usize;
                        let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                        let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                        let dpn = dn - hdmin[v][jpv];
                        let mut dp_v = dpn;
                        let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                        let bv = rb(v, jpv);
                        let by = rb(y, jpysdr);
                        while dp_v <= dx - hdmin[v][jpv] {
                            if do_j_v && do_j_y {
                                ja[bv + dp_v as usize] = flogsum(ja[bv + dp_v as usize], ja[by + dp_y_sd as usize] + tsc);
                            }
                            if do_l_v && do_l_y {
                                la[bv + dp_v as usize] = flogsum(la[bv + dp_v as usize], la[by + dp_y_sd as usize] + tsc);
                            }
                            if do_r_v && do_r_y {
                                ra[bv + dp_v as usize] = flogsum(ra[bv + dp_v as usize], ra[by + dp_y_sd as usize] + tsc);
                            }
                            // d == 0: force L and R to IMPOSSIBLE (C:3140-3144) — unchanged
                            if dp_v == dpn && dn == 0 {
                                if do_l_v { la[bv + dp_v as usize] = ninf; }
                                if do_r_v { ra[bv + dp_v as usize] = ninf; }
                            }
                            dp_v += 1;
                            dp_y_sd += 1;
                        }
                        jp_v += 1;
                        jp_y_sdr += 1;
                    }
                }
            }
        } else {
            // B_st (C:3151-3276) — FLogsum for all k-combine + child-only inherit loops
            let y = cfirst; // left subtree (BEGL_S)
            let z = cnum; // right subtree (BEGR_S)
            let do_j_y = jvalid[y];
            let do_l_y = lvalid[y] && fill_l;
            let do_r_y = rvalid[y] && fill_r;
            let do_j_z = jvalid[z];
            let do_l_z = lvalid[z] && fill_l;
            let do_r_z = rvalid[z] && fill_r;

            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let hdmin_zjp = hdmin[z][jp_z];
                let kn = (j - jmax[y]).max(hdmin_zjp).max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                let base_v = rb(v, jp_v);
                let base_z = rb(z, jp_z);
                let hdmin_vjp = hdmin[v][jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax[v][jp_v] {
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let jp_y_mk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jp_y_mk] && k <= d - hdmin[y][jp_y_mk] {
                            let kp_z = (k - hdmin_zjp) as usize;
                            let dp_y = d - hdmin[y][jp_y_mk];
                            let cy = rb(y, jp_y_mk) + (dp_y - k) as usize;
                            let cz = base_z + kp_z;
                            if do_j_v && do_j_y && do_j_z {
                                ja[base_v + dp_v] = flogsum(ja[base_v + dp_v], ja[cy] + ja[cz]);
                            }
                            if do_l_v && do_j_y && do_l_z {
                                la[base_v + dp_v] = flogsum(la[base_v + dp_v], ja[cy] + la[cz]);
                            }
                            if do_r_v && do_r_y && do_j_z {
                                ra[base_v + dp_v] = flogsum(ra[base_v + dp_v], ra[cy] + ja[cz]);
                            }
                            if k != 0 && k != d && do_t_v && do_r_y && do_l_z {
                                ta[base_v + dp_v] = flogsum(ta[base_v + dp_v], ra[cy] + la[cz]);
                            }
                        }
                        k += 1;
                    }
                    d += 1;
                }
            }
            // special case L: Lalpha[v] from y (J or L), independent of z (C:3237-3256)
            if do_l_v && (do_j_y || do_l_y) {
                let jn = jmin[v].max(jmin[y]);
                let jx = jmax[v].min(jmax[y]);
                for j in jn..=jx {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_y = (j - jmin[y]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    let base_v = rb(v, jp_v);
                    let base_y = rb(y, jp_y);
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        if do_j_y {
                            la[base_v + dp_v] = flogsum(la[base_v + dp_v], ja[base_y + dp_y]);
                        }
                        if do_l_y {
                            la[base_v + dp_v] = flogsum(la[base_v + dp_v], la[base_y + dp_y]);
                        }
                        d += 1;
                    }
                }
            }
            // special case R: Ralpha[v] from z (J or R), independent of y (C:3257-3276)
            if do_r_v && (do_j_z || do_r_z) {
                let jn = jmin[v].max(jmin[z]);
                let jx = jmax[v].min(jmax[z]);
                for j in jn..=jx {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_z = (j - jmin[z]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[z][jp_z]);
                    let dx = hdmax[v][jp_v].min(hdmax[z][jp_z]);
                    let base_v = rb(v, jp_v);
                    let base_z = rb(z, jp_z);
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_z = (d - hdmin[z][jp_z]) as usize;
                        if do_j_z {
                            ra[base_v + dp_v] = flogsum(ra[base_v + dp_v], ja[base_z + dp_z]);
                        }
                        if do_r_z {
                            ra[base_v + dp_v] = flogsum(ra[base_v + dp_v], ra[base_z + dp_z]);
                        }
                        d += 1;
                    }
                }
            }
        }
    }

    // ---- ROOT_S (v=0): truncated begins + hit reporting (C:3352-3448) ----
    // NOTE: the truncated-begin update STAYS MAX (`if sc > {J,L,R,T}alpha[0]`), NOT
    // FLogsum (C:3383,3398,3413,3428) — identical to the CYK ROOT block.
    let do_j_0 = jvalid[0];
    let do_l_0 = lvalid[0] && fill_l;
    let do_r_0 = rvalid[0] && fill_r;
    let do_t_0 = tvalid[0] && fill_t;

    let mut tmp_hits: Vec<TrHit> = Vec::new();
    let mut bestr = vec![0i32; ww + 1];
    let mut bestsc = vec![IMPOSSIBLE; ww + 1];
    let mut bestmode = vec![TRMODE_UNKNOWN; ww + 1];

    for j in jmin[0]..=jmax[0] {
        let jp_v = (j - jmin[0]) as usize;
        let base_v = rb(0, jp_v);
        for d in 0..=ww {
            bestr[d] = 0;
            bestsc[d] = IMPOSSIBLE;
            bestmode[d] = TRMODE_UNKNOWN;
        }
        for y in 1..m {
            let trpenalty = pty[y];
            if not_impossible_f(trpenalty) && j >= jmin[y] && j <= jmax[y] {
                let do_j_y = jvalid[y];
                let do_l_y = lvalid[y] && fill_l;
                let do_r_y = rvalid[y] && fill_r;
                let do_t_y = tvalid[y] && fill_t;
                let jp_y = (j - jmin[y]) as usize;
                let dn = hdmin[0][jp_v].max(hdmin[y][jp_y]);
                let dx = hdmax[0][jp_v].min(hdmax[y][jp_y]);
                let cy = rb(y, jp_y);
                if do_j_0 && do_j_y {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = ja[cy + dp_y] + trpenalty;
                        if sc > ja[base_v + dp_v] {
                            ja[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_J;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
                if do_l_0 && do_l_y {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = la[cy + dp_y] + trpenalty;
                        if sc > la[base_v + dp_v] {
                            la[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_L;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
                if do_r_0 && do_r_y {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = ra[cy + dp_y] + trpenalty;
                        if sc > ra[base_v + dp_v] {
                            ra[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_R;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
                if do_t_0 && do_t_y && cm.sttype[y] as i32 == B_ST {
                    let mut d = dn;
                    while d <= dx {
                        let dp_v = (d - hdmin[0][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        let sc = ta[cy + dp_y] + trpenalty;
                        if sc > ta[base_v + dp_v] {
                            ta[base_v + dp_v] = sc;
                            if sc > bestsc[d as usize] {
                                bestsc[d as usize] = sc;
                                bestmode[d as usize] = TRMODE_T;
                                bestr[d as usize] = y as i32;
                            }
                        }
                        d += 1;
                    }
                }
            }
        }
        // report hits greedily to tmp_hits (C ReportHitsGreedily branch)
        report_hits_greedily(
            cm, pass_idx, local, j, hdmin[0][jp_v], hdmax[0][jp_v], &bestsc, &bestr, &bestmode,
            ww, if do_null3 { Some(&act) } else { None }, i0, j0, cutoff, &mut tmp_hits,
        );
    }


    // ---- find best scoring hit + envelope boundaries (C:3459-3501) ----
    // best-over-4-planes STAYS MAX (`> vsc_root`), NOT FLogsum (C:3470-3485).
    let mut vsc_root = IMPOSSIBLE;
    let mut vmode_root = TRMODE_UNKNOWN;
    let jpx = jmax[0] - jmin[0];
    for jp_v_i in 0..=jpx.max(-1) {
        if jmax[0] < jmin[0] { break; }
        let jp_v = jp_v_i as usize;
        let j = jp_v as i32 + jmin[0];
        let base_v = rb(0, jp_v);
        let dpx = hdmax[0][jp_v] - hdmin[0][jp_v];
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let c = base_v + dp_v as usize;
            if do_j_0 && ja[c] > vsc_root { vsc_root = ja[c]; vmode_root = TRMODE_J; }
            if do_l_0 && la[c] > vsc_root { vsc_root = la[c]; vmode_root = TRMODE_L; }
            if do_r_0 && ra[c] > vsc_root { vsc_root = ra[c]; vmode_root = TRMODE_R; }
            if do_t_0 && ta[c] > vsc_root { vsc_root = ta[c]; vmode_root = TRMODE_T; }
        }
        // envelope
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let c = base_v + dp_v as usize;
            if (do_j_0 && ja[c] >= env_cutoff)
                || (do_l_0 && la[c] >= env_cutoff)
                || (do_r_0 && ra[c] >= env_cutoff)
                || (do_t_0 && ta[c] >= env_cutoff)
            {
                let i = j - (dp_v + hdmin[0][jp_v]) + 1;
                if (i as i64) < envi { envi = i as i64; }
                if (j as i64) > envj { envj = j as i64; }
            }
        }
    }

    let ret_envi = if envi == (j0 + 1) as i64 { -1 } else { envi };
    let ret_envj = if envj == (i0 - 1) as i64 { -1 } else { envj };

    // greedy overlap removal (C: SortForOverlapRemoval + RemoveOrMarkOverlaps)
    let kept = remove_overlaps_greedy_tr(tmp_hits);
    (kept, vsc_root, vmode_root, ret_envi, ret_envj)
}

// ============================================================================
// Generic truncated scanner.
// ============================================================================
#[allow(clippy::too_many_arguments)]
fn generic_tr_scan<S: TrSr>(
    cm: &CM,
    tab: &TrTables<S>,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    cutoff: f32,
    do_null3: bool,
    qdbidx: usize,
    pass_idx: i32,
    local: bool,
) -> (Vec<TrHit>, f32, i8, f32) {
    let m = cm.m as usize;
    let kp = ALPHABET_SIZE_P;
    let smx_w = cm.w;
    let l = j0 - i0 + 1;
    let mut w = smx_w;
    if w > l {
        w = l;
    }
    let w = w as usize;

    let (fill_l, fill_r, fill_t) = fill_from_pass_idx(pass_idx);

    // dmax source (only used for B-state k-band and dxcap when banded)
    let full_dmax: Vec<i32>;
    let nonbanded = qdbidx == SMX_NOQDB;
    let dmax: &[i32] = if nonbanded {
        full_dmax = vec![smx_w; m];
        &full_dmax
    } else if qdbidx == SMX_QDB1_TIGHT {
        &cm.dmax1
    } else {
        &cm.dmax2
    };

    // Truncated bands (cm_mx.c:6429): dn = 1 for every state; dxcap = min(dmax,W).
    let mut dxcap_v = vec![0i32; m];
    for v in 0..m {
        dxcap_v[v] = dmax[v].min(smx_w);
    }
    let dn_all = 1i32.min(smx_w).max(1);

    // BEGL_S compact indexing.
    let mut begl_idx = vec![usize::MAX; m];
    let mut nbegl = 0usize;
    for v in 0..m {
        if cm.stid[v] as i32 == BEGL_S {
            begl_idx[v] = nbegl;
            nbegl += 1;
        }
    }

    // init_scAA[v][d] = endsc valid ? el*d + endsc : zero (FCalcInitDPScores).
    let mut el_sca = vec![S::base0(); w + 1];
    for d in 1..=w {
        el_sca[d] = S::addv(el_sca[d - 1], tab.el_self);
    }
    let mut init_sc = vec![vec![S::zero(); w + 1]; m];
    for v in 0..m {
        let es = tab.endsc[v];
        if !S::is_impossible(es) {
            for d in 0..=w {
                init_sc[v][d] = S::addv(el_sca[d], es);
            }
        }
    }

    // Four marginal planes for non-BEGL states: [2][M][W+1]. T only meaningful for
    // B states but allocated uniformly. BEGL decks: [W+1][nbegl][W+1] (J/L/R).
    let z = S::zero();
    let mut ja = vec![vec![vec![z; w + 1]; m]; 2];
    let mut la = vec![vec![vec![z; w + 1]; m]; 2];
    let mut ra = vec![vec![vec![z; w + 1]; m]; 2];
    let mut ta = vec![vec![vec![z; w + 1]; m]; 2];
    let mut jbegl = vec![vec![vec![z; w + 1]; nbegl]; w + 1];
    let mut lbegl = vec![vec![vec![z; w + 1]; nbegl]; w + 1];
    let mut rbegl = vec![vec![vec![z; w + 1]; nbegl]; w + 1];

    // ---- d=0 base cases (cm_tr_scan_mx_Initialize*), done ONCE ----
    for v in (0..m).rev() {
        if cm.stid[v] as i32 != BEGL_S {
            let stt = cm.sttype[v] as i32;
            if stt == E_ST {
                ja[0][v][0] = S::base0();
                ja[1][v][0] = S::base0();
                la[0][v][0] = S::base0();
                la[1][v][0] = S::base0();
                ra[0][v][0] = S::base0();
                ra[1][v][0] = S::base0();
            } else if stt == S_ST || stt == D_ST {
                let y = cm.cfirst[v] as usize;
                let mut a0 = tab.endsc[v];
                for yo in 0..cm.cnum[v] as usize {
                    a0 = S::max2(a0, S::addv(ja[0][y + yo][0], tab.tsc[v][yo]));
                }
                a0 = S::max2(a0, S::zero());
                ja[0][v][0] = a0;
                ja[1][v][0] = a0;
            } else if stt == B_ST {
                let wl = cm.cfirst[v] as usize;
                let y = cm.cnum[v] as usize;
                let a0 = S::addv(jbegl[0][begl_idx[wl]][0], ja[0][y][0]);
                ja[0][v][0] = a0;
                ja[1][v][0] = a0;
            } else {
                // emitters: J[*][v][0] stays IMPOSSIBLE
            }
        } else {
            let bi = begl_idx[v];
            let y = cm.cfirst[v] as usize;
            let mut a0 = tab.endsc[v];
            for yo in 0..cm.cnum[v] as usize {
                a0 = S::max2(a0, S::addv(ja[0][y + yo][0], tab.tsc[v][yo]));
            }
            a0 = S::max2(a0, S::zero());
            for j in 0..=w {
                jbegl[j][bi][0] = a0;
            }
        }
    }

    // null3 act vector
    let mut act: Vec<[f64; 4]> = if do_null3 { vec![[0.0; 4]; w + 1] } else { Vec::new() };

    let mut jp_wa = vec![0usize; w + 1];
    let mut bestsc = vec![IMPOSSIBLE; w + 1];
    let mut bestr = vec![0i32; w + 1];
    let mut bestmode = vec![TRMODE_UNKNOWN; w + 1];

    let mut vsc_root = IMPOSSIBLE;
    let mut vmode_root = TRMODE_UNKNOWN;
    let mut bsc_full = IMPOSSIBLE;

    let mut tmp_hits: Vec<TrHit> = Vec::new();

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
            let is_begl = cm.stid[v] as i32 == BEGL_S;
            let jp_v = if is_begl { (j.rem_euclid(w as i32 + 1)) as usize } else { cur };
            let jp_y = if state_right_delta(stt) > 0 { prv } else { cur };
            let jq_y = if state_right_delta(stt) > 0 { cur } else { prv };
            let dn = dn_all;
            let dx = (jrow as i32).min(dxcap_v[v]);
            let esc_v: &[S] = &tab.oesc[v];
            let tsc_v = &tab.tsc[v];
            let lmesc_v: &[S] = &tab.lmesc[v];
            let rmesc_v: &[S] = &tab.rmesc[v];
            let esc_j = if stt == IR_ST || stt == MR_ST {
                esc_v[dsq[j as usize] as usize]
            } else {
                S::zero()
            };
            let rmesc_j = if stt == IR_ST || stt == MR_ST || stt == MP_ST {
                rmesc_v[dsq[j as usize] as usize]
            } else {
                S::zero()
            };

            if stt == B_ST {
                let wl = cm.cfirst[v] as usize; // BEGL_S
                let y = cm.cnum[v] as usize; // BEGR_S
                let bi = begl_idx[wl];
                let dx_y = dmax[y].min(smx_w);
                let dx_w = dmax[wl].min(smx_w);
                for d in dn..=dx {
                    let (kmin, kmax) = if !nonbanded {
                        (0.max(d - dx_w), dx_y.min(d))
                    } else {
                        (0, d)
                    };
                    let mut jsc = init_sc[v][(d - sd) as usize];
                    let mut lsc = if fill_l { init_sc[v][(d - sd) as usize] } else { z };
                    let mut rsc = if fill_r { init_sc[v][(d - sd) as usize] } else { z };
                    let mut tsc = if fill_t { init_sc[v][(d - sd) as usize] } else { z };
                    let mut k = kmin;
                    while k <= kmax {
                        let lft = jbegl[jp_wa[k as usize]][bi][(d - k) as usize];
                        jsc = S::comb(jsc, S::addv(lft, ja[jp_y][y][k as usize]));
                        if fill_l {
                            lsc = S::comb(lsc, S::addv(lft, la[jp_y][y][k as usize]));
                        }
                        if fill_r {
                            let lftr = rbegl[jp_wa[k as usize]][bi][(d - k) as usize];
                            rsc = S::comb(rsc, S::addv(lftr, ja[jp_y][y][k as usize]));
                        }
                        k += 1;
                    }
                    if fill_t {
                        let kn = 1.max(kmin);
                        let kx = (d - 1).min(kmax);
                        let mut k = kn;
                        while k <= kx {
                            let lftr = rbegl[jp_wa[k as usize]][bi][(d - k) as usize];
                            tsc = S::comb(tsc, S::addv(lftr, la[jp_y][y][k as usize]));
                            k += 1;
                        }
                    }
                    ja[jp_v][v][d as usize] = jsc;
                    if fill_t {
                        ta[jp_v][v][d as usize] = tsc;
                    }
                    if fill_l {
                        la[jp_v][v][d as usize] = if kmin == 0 {
                            S::comb(
                                lsc,
                                S::max2(jbegl[jp_wa[0]][bi][d as usize], lbegl[jp_wa[0]][bi][d as usize]),
                            )
                        } else {
                            lsc
                        };
                    }
                    if fill_r {
                        ra[jp_v][v][d as usize] = if kmax == d {
                            S::comb(rsc, S::max2(ja[jp_y][y][d as usize], ra[jp_y][y][d as usize]))
                        } else {
                            rsc
                        };
                    }
                }
            } else if is_begl {
                let y = cm.cfirst[v] as usize;
                let bi = begl_idx[v];
                for d in dn..=dx {
                    let mut jsc = init_sc[v][(d - sd) as usize];
                    let mut lsc = if fill_l { init_sc[v][(d - sd) as usize] } else { z };
                    let mut rsc = if fill_r { init_sc[v][(d - sd) as usize] } else { z };
                    for yo in 0..cm.cnum[v] as usize {
                        jsc = S::comb(jsc, S::addv(ja[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                        if fill_l {
                            lsc = S::comb(lsc, S::addv(la[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                        }
                        if fill_r {
                            rsc = S::comb(rsc, S::addv(ra[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                        }
                    }
                    jbegl[jp_v][bi][d as usize] = jsc;
                    if fill_l {
                        lbegl[jp_v][bi][d as usize] = lsc;
                    }
                    if fill_r {
                        rbegl[jp_v][bi][d as usize] = rsc;
                    }
                }
            } else if em == EMITLEFT {
                if !state_is_detached(cm, v) {
                    let y = cm.cfirst[v] as usize;
                    let ryoffset0 = if stt == IL_ST { 1usize } else { 0usize };
                    let mut i = j - dn + 1;
                    for d in dn..=dx {
                        let mut jsc = init_sc[v][(d - sd) as usize];
                        let mut lsc = if fill_l { init_sc[v][(d - sd) as usize] } else { z };
                        if fill_r {
                            // 'd', not 'd-sd' (won't emit left in R mode); pre-store.
                            let rsc0 = init_sc[v][d as usize];
                            ra[jp_v][v][d as usize] = rsc0;
                        }
                        for yo in 0..cm.cnum[v] as usize {
                            jsc = S::comb(jsc, S::addv(ja[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                            if fill_l {
                                lsc = S::comb(lsc, S::addv(la[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                            }
                        }
                        let e = esc_v[dsq[i as usize] as usize];
                        ja[jp_v][v][d as usize] = S::addv(jsc, e);
                        if fill_l {
                            la[jp_v][v][d as usize] = if d >= 2 { S::addv(lsc, e) } else { e };
                        }
                        if fill_r {
                            let mut rsc = ra[jp_v][v][d as usize];
                            for yo in ryoffset0..cm.cnum[v] as usize {
                                rsc = S::comb(
                                    rsc,
                                    S::max2(
                                        S::addv(ja[jp_y][y + yo][d as usize], tsc_v[yo]),
                                        S::addv(ra[jp_y][y + yo][d as usize], tsc_v[yo]),
                                    ),
                                );
                            }
                            ra[jp_v][v][d as usize] = rsc;
                        }
                        i -= 1;
                    }
                }
            } else if em == EMITRIGHT {
                if !state_is_detached(cm, v) {
                    let y = cm.cfirst[v] as usize;
                    let lyoffset0 = if stt == IR_ST { 1usize } else { 0usize };
                    for d in dn..=dx {
                        let mut jsc = init_sc[v][(d - sd) as usize];
                        let mut rsc = if fill_r { init_sc[v][(d - sd) as usize] } else { z };
                        if fill_l {
                            let lsc0 = init_sc[v][d as usize];
                            la[jp_v][v][d as usize] = lsc0;
                        }
                        for yo in 0..cm.cnum[v] as usize {
                            jsc = S::comb(jsc, S::addv(ja[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                            if fill_r {
                                rsc = S::comb(rsc, S::addv(ra[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                            }
                        }
                        ja[jp_v][v][d as usize] = S::addv(jsc, esc_j);
                        if fill_r {
                            ra[jp_v][v][d as usize] = if d >= 2 { S::addv(rsc, esc_j) } else { esc_j };
                        }
                        if fill_l {
                            let mut lsc = la[jp_v][v][d as usize];
                            for yo in lyoffset0..cm.cnum[v] as usize {
                                lsc = S::comb(
                                    lsc,
                                    S::max2(
                                        S::addv(ja[jq_y][y + yo][d as usize], tsc_v[yo]),
                                        S::addv(la[jq_y][y + yo][d as usize], tsc_v[yo]),
                                    ),
                                );
                            }
                            la[jp_v][v][d as usize] = lsc;
                        }
                    }
                }
            } else if em == EMITPAIR {
                let y = cm.cfirst[v] as usize;
                let mut i = j - dn + 1;
                for d in dn..=dx {
                    // J needs d-2 (d-sd); for d==1 C reads the adjacent init/alpha
                    // cell but the result is discarded by the `d>=2 ? : IMPOSSIBLE`
                    // guard below. We guard the d-2 accesses to avoid a Rust OOB while
                    // producing the identical (discarded) result.
                    let mut jsc = if d >= 2 { init_sc[v][(d - sd) as usize] } else { z };
                    let mut lsc = if fill_l { init_sc[v][(d - 1) as usize] } else { z };
                    let mut rsc = if fill_r { init_sc[v][(d - 1) as usize] } else { z };
                    for yo in 0..cm.cnum[v] as usize {
                        if d >= 2 {
                            jsc = S::comb(jsc, S::addv(ja[jp_y][y + yo][(d - 2) as usize], tsc_v[yo]));
                        }
                        if fill_l {
                            lsc = S::comb(
                                lsc,
                                S::max2(
                                    S::addv(ja[jq_y][y + yo][(d - 1) as usize], tsc_v[yo]),
                                    S::addv(la[jq_y][y + yo][(d - 1) as usize], tsc_v[yo]),
                                ),
                            );
                        }
                        if fill_r {
                            rsc = S::comb(
                                rsc,
                                S::max2(
                                    S::addv(ja[jp_y][y + yo][(d - 1) as usize], tsc_v[yo]),
                                    S::addv(ra[jp_y][y + yo][(d - 1) as usize], tsc_v[yo]),
                                ),
                            );
                        }
                    }
                    let idx_pair = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                    ja[jp_v][v][d as usize] = if d >= 2 { S::addv(jsc, esc_v[idx_pair]) } else { S::zero() };
                    if fill_l {
                        let lm = lmesc_v[dsq[i as usize] as usize];
                        la[jp_v][v][d as usize] = if d >= 2 { S::addv(lsc, lm) } else { lm };
                    }
                    if fill_r {
                        ra[jp_v][v][d as usize] = if d >= 2 { S::addv(rsc, rmesc_j) } else { rmesc_j };
                    }
                    i -= 1;
                }
            } else {
                // EMITNONE (D, S)
                let y = cm.cfirst[v] as usize;
                for d in dn..=dx {
                    let mut jsc = init_sc[v][(d - sd) as usize];
                    let mut lsc = if fill_l { init_sc[v][(d - sd) as usize] } else { z };
                    let mut rsc = if fill_r { init_sc[v][(d - sd) as usize] } else { z };
                    for yo in 0..cm.cnum[v] as usize {
                        jsc = S::comb(jsc, S::addv(ja[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                        if fill_l {
                            lsc = S::comb(lsc, S::addv(la[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                        }
                        if fill_r {
                            rsc = S::comb(rsc, S::addv(ra[jp_y][y + yo][(d - sd) as usize], tsc_v[yo]));
                        }
                    }
                    ja[jp_v][v][d as usize] = jsc;
                    if fill_l {
                        la[jp_v][v][d as usize] = lsc;
                    }
                    if fill_r {
                        ra[jp_v][v][d as usize] = rsc;
                    }
                }
            }
        }

        // ---- ROOT_S (v=0): truncated begins. Persistent-row idiom (ja[cur][0][d]
        // is never reset per-j). ----
        // dn/dx for ROOT (v=0): dn=1, dx=min(jrow, dxcap[0]).
        let dn0 = dn_all;
        let dx0 = (jrow as i32).min(dxcap_v[0]);
        for d in 0..=w {
            bestr[d] = 0;
            bestsc[d] = IMPOSSIBLE;
            bestmode[d] = TRMODE_UNKNOWN;
        }
        for y in 1..m {
            let trpenalty = tab.pty[y];
            if S::is_impossible(trpenalty) {
                continue;
            }
            let dn = dn0.max(dn_all); // dnA[0].max(dnA[y]) = 1
            let dx = dx0.min((jrow as i32).min(dxcap_v[y]));
            // J
            for d in dn..=dx {
                let sc = S::addv(ja[cur][y][d as usize], trpenalty);
                if S::ord(sc) > S::ord(ja[cur][0][d as usize]) {
                    ja[cur][0][d as usize] = sc;
                    let fsc = S::scbits(sc);
                    if fsc > bestsc[d as usize] {
                        bestsc[d as usize] = fsc;
                        bestmode[d as usize] = TRMODE_J;
                        bestr[d as usize] = y as i32;
                    }
                }
            }
            if fill_l {
                for d in dn..=dx {
                    let sc = S::addv(la[cur][y][d as usize], trpenalty);
                    if S::ord(sc) > S::ord(la[cur][0][d as usize]) {
                        la[cur][0][d as usize] = sc;
                        let fsc = S::scbits(sc);
                        if fsc > bestsc[d as usize] {
                            bestsc[d as usize] = fsc;
                            bestmode[d as usize] = TRMODE_L;
                            bestr[d as usize] = y as i32;
                        }
                    }
                }
            }
            if fill_r {
                for d in dn..=dx {
                    let sc = S::addv(ra[cur][y][d as usize], trpenalty);
                    if S::ord(sc) > S::ord(ra[cur][0][d as usize]) {
                        ra[cur][0][d as usize] = sc;
                        let fsc = S::scbits(sc);
                        if fsc > bestsc[d as usize] {
                            bestsc[d as usize] = fsc;
                            bestmode[d as usize] = TRMODE_R;
                            bestr[d as usize] = y as i32;
                        }
                    }
                }
            }
            if fill_t && cm.sttype[y] as i32 == B_ST {
                for d in dn..=dx {
                    let sc = S::addv(ta[cur][y][d as usize], trpenalty);
                    if S::ord(sc) > S::ord(ta[cur][0][d as usize]) {
                        ta[cur][0][d as usize] = sc;
                        let fsc = S::scbits(sc);
                        if fsc > bestsc[d as usize] {
                            bestsc[d as usize] = fsc;
                            bestmode[d as usize] = TRMODE_T;
                            bestr[d as usize] = y as i32;
                        }
                    }
                }
            }
        }

        // update vsc_root / vmode_root
        for d in dn0..=dx0 {
            if bestsc[d as usize] > vsc_root {
                vsc_root = bestsc[d as usize];
                vmode_root = bestmode[d as usize];
            }
        }
        // best score spanning the full window (C bsc_full: bestsc[j] at j==j0)
        if j == j0 {
            let di = j as usize;
            if di <= w && bestsc[di] > bsc_full {
                bsc_full = bestsc[di];
            }
        }

        // report hits greedily (ReportHitsGreedily + AllowTruncation)
        report_hits_greedily(
            cm, pass_idx, local, j, dn0, dx0, &bestsc, &bestr, &bestmode, w,
            if do_null3 { Some(&act) } else { None }, i0, j0, cutoff, &mut tmp_hits,
        );
    }

    // ---- greedy overlap removal (SortForOverlapRemoval + RemoveOrMarkOverlaps) ----
    let kept = remove_overlaps_greedy_tr(tmp_hits);
    let _ = (fill_t, PLI_PASS_STD_ANY, BIF_B);
    (kept, vsc_root, vmode_root, bsc_full)
}

// ============================================================================
// Truncated ReportHitsGreedily (cm_mx.c:7555) + cm_hit_AllowTruncation.
// ============================================================================
#[allow(clippy::too_many_arguments)]
fn report_hits_greedily(
    cm: &CM,
    pass_idx: i32,
    local: bool,
    j: i32,
    dmin: i32,
    dmax: i32,
    bestsc: &[f32],
    bestr: &[i32],
    bestmode: &[i8],
    w: usize,
    act: Option<&Vec<[f64; 4]>>,
    i0: i32,
    j0: i32,
    cutoff: f32,
    out: &mut Vec<TrHit>,
) {
    if dmin > dmax {
        return;
    }
    let mut max_reported = IMPOSSIBLE;
    let dlo = dmin.max(1);
    for d in dlo..=dmax {
        let i = j - d + 1;
        let mode = bestmode[d as usize];
        let mut hit_sc = bestsc[d as usize];
        let mut bias = 0.0f32;
        if hit_sc > max_reported && hit_sc >= cutoff && not_impossible_f(hit_sc) {
            let mut do_report = allow_truncation(cm, pass_idx, local, i as i64, j as i64, i0 as i64, j0 as i64, mode, bestr[d as usize]);
            if do_report {
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
            }
            if do_report {
                out.push(TrHit { i, j, score: hit_sc, bias, mode, root: bestr[d as usize] });
                max_reported = hit_sc;
            }
        }
    }
}

/// C `cm_hit_AllowTruncation` (cm_tophits.c:3053).
#[allow(clippy::too_many_arguments)]
fn allow_truncation(
    cm: &CM,
    pass_idx: i32,
    local: bool,
    start: i64,
    stop: i64,
    i0: i64,
    j0: i64,
    mode: i8,
    b: i32,
) -> bool {
    use crate::cm_trunc::PLI_PASS_5P_AND_3P_ANY;
    // PLI_PASS_STD_ANY / 5P_AND_3P_ANY / HMM_ONLY_ANY: allow all hits.
    if pass_idx == PLI_PASS_STD_ANY || pass_idx == PLI_PASS_5P_AND_3P_ANY {
        return true;
    }
    if start == i0 && stop == j0 {
        return true;
    }
    let nd = cm.ndidx[b as usize] as usize;
    let ndt = cm.ndtype[nd] as i32;
    let lpos = if ndt == MATP_ND || ndt == MATL_ND {
        cm_emap_lpos(cm, nd)
    } else {
        cm_emap_lpos(cm, nd) + 1
    };
    let rpos = if ndt == MATP_ND || ndt == MATR_ND {
        cm_emap_rpos(cm, nd)
    } else {
        cm_emap_rpos(cm, nd) - 1
    };
    if local {
        match mode {
            TRMODE_J => true,
            TRMODE_L => stop == j0,
            TRMODE_R => start == i0,
            _ => false,
        }
    } else {
        match mode {
            TRMODE_J => lpos == 1 && rpos == cm.clen,
            TRMODE_L => stop == j0 && lpos == 1,
            TRMODE_R => start == i0 && rpos == cm.clen,
            _ => false,
        }
    }
}

// emap lpos/rpos accessors (built lazily onto the CM via cp9::create_emit_map).
// We recompute here from the emit map stored on demand; to avoid recomputing per hit
// the caller-side wiring passes an already-built emap through the CM's helper.
thread_local! {
    static EMAP_CACHE: std::cell::RefCell<Option<(usize, crate::cp9::EmitMap)>> =
        const { std::cell::RefCell::new(None) };
}
fn cm_emap_lpos(cm: &CM, nd: usize) -> i32 {
    EMAP_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let need = c.as_ref().map(|(m, _)| *m != cm.m as usize).unwrap_or(true);
        if need {
            *c = Some((cm.m as usize, crate::cp9::create_emit_map(cm)));
        }
        c.as_ref().unwrap().1.lpos[nd]
    })
}
fn cm_emap_rpos(cm: &CM, nd: usize) -> i32 {
    EMAP_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let need = c.as_ref().map(|(m, _)| *m != cm.m as usize).unwrap_or(true);
        if need {
            *c = Some((cm.m as usize, crate::cp9::create_emit_map(cm)));
        }
        c.as_ref().unwrap().1.rpos[nd]
    })
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

/// Greedy overlap removal over TrHit (C cm_tophits_SortForOverlapRemoval +
/// RemoveOrMarkOverlaps): sort by score desc then start asc; drop any later hit
/// that overlaps a kept one.
fn remove_overlaps_greedy_tr(mut hits: Vec<TrHit>) -> Vec<TrHit> {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.i.cmp(&b.i))
    });
    let n = hits.len();
    let mut removed = vec![false; n];
    for i in 0..n {
        if removed[i] {
            continue;
        }
        let (si, ei) = (hits[i].i, hits[i].j);
        for k in (i + 1)..n {
            if removed[k] {
                continue;
            }
            let (sk, ek) = (hits[k].i, hits[k].j);
            if !(ek < si) && !(ei < sk) {
                removed[k] = true;
            }
        }
    }
    hits.into_iter().zip(removed).filter(|(_, r)| !*r).map(|(h, _)| h).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cm_trunc::{
        tr_cyk_align, TrPenalties, PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE,
        PLI_PASS_5P_AND_3P_FORCE,
    };

    const CM_PATH: &str = "/mnt/DAS/sunju/programme/bactars/infernal/original/tutorial/tRNA5.c.cm";

    fn digitize(seq: &str) -> Vec<u8> {
        // dsq: [255, r1..rL, 255]; ACGU/T -> 0,1,2,3.
        let mut d = vec![255u8];
        for c in seq.chars() {
            let r = match c.to_ascii_uppercase() {
                'A' => 0u8,
                'C' => 1,
                'G' => 2,
                'U' | 'T' => 3,
                _ => continue,
            };
            d.push(r);
        }
        d.push(255);
        d
    }

    fn setup() -> (CM, TrPenalties) {
        let mut cm = crate::cm_file::cm_file_read_global(CM_PATH).expect("read cm");
        crate::cm_nohmm::cm_configure_scores_global(&mut cm);
        crate::cm_nohmm::cm_calc_qdb_bands(&mut cm, 1e-7, 1e-15, 1e-7).expect("qdb");
        let emap = crate::cp9::create_emit_map(&cm);
        let psi = crate::cp9::cm_expected_state_occupancy(&cm);
        let trp = TrPenalties::new(&cm, &emap, &psi);
        (cm, trp)
    }

    // The float truncated CYK SCAN's full-window score (bsc_full) must equal the
    // C-verified whole-sequence truncated CYK ALIGN score (tr_cyk_align), byte-for-
    // byte, for each truncated pass on a canonical sequence.
    #[test]
    fn scan_fullwindow_equals_align_cyk() {
        let (cm, trp) = setup();
        let seq = "gCcggcAUAGcgcAgUGGuAgcgCgccagccUgucAagcuggAGgUCCgggGUUCGAUUCcccGUgccgGca";
        let dsq = digitize(seq);
        let l = (dsq.len() - 2) as i32;
        let lm = cm.lmesc.clone();
        let rm = cm.rmesc.clone();
        for pass in [
            PLI_PASS_5P_ONLY_FORCE,
            PLI_PASS_3P_ONLY_FORCE,
            PLI_PASS_5P_AND_3P_FORCE,
        ] {
            let (_tr, align_sc, _mode) = tr_cyk_align(&cm, &trp, &lm, &rm, &dsq, l, pass, false);
            let (_hits, vsc_root, _vmode, bsc_full) =
                ref_tr_cyk_scan(&cm, &trp, &dsq, 1, l, -1.0e30, false, SMX_NOQDB, pass, false);
            eprintln!(
                "pass {pass}: align_sc={align_sc:.6} scan_bsc_full={bsc_full:.6} vsc_root={vsc_root:.6}"
            );
            assert!(
                (align_sc - bsc_full).abs() < 1e-4,
                "pass {pass}: full-window scan score {bsc_full} != align score {align_sc}"
            );
            assert!(
                vsc_root >= bsc_full - 1e-4,
                "vsc_root must be >= full-window score"
            );
        }
    }

    #[test]
    fn scan_qdb_runs() {
        let (cm, trp) = setup();
        let seq = "gCcggcAUAGcgcAgUGGuAgcgCgccagccUgucAagcuggAGgUCCgggGUUCGAUUCcccGUgccgGca";
        let dsq = digitize(seq);
        let l = (dsq.len() - 2) as i32;
        let (_h, vsc_nb, _m, _b) = ref_tr_cyk_scan(
            &cm, &trp, &dsq, 1, l, -1.0e30, false, SMX_NOQDB, PLI_PASS_5P_AND_3P_FORCE, false,
        );
        let (_h2, vsc_q, _m2, _b2) = ref_tr_cyk_scan(
            &cm, &trp, &dsq, 1, l, -1.0e30, false, SMX_QDB1_TIGHT, PLI_PASS_5P_AND_3P_FORCE, false,
        );
        eprintln!("nonbanded vsc={vsc_nb:.4} qdb1 vsc={vsc_q:.4}");
        assert!(vsc_q <= vsc_nb + 1e-4, "QDB score must not exceed non-banded");
    }

    // The integer truncated Inside SCAN must run and produce a score >= the CYK scan
    // score for the same pass/window (Inside sums over parses, CYK maxes), exercising
    // RefITrInsideScan (STEP 2) which shares the recursion structure with the CYK scan.
    #[test]
    fn inside_scan_ge_cyk_scan() {
        let (cm, trp) = setup();
        let seq = "gCcggcAUAGcgcAgUGGuAgcgCgccagccUgucAagcuggAGgUCCgggGUUCGAUUCcccGUgccgGca";
        let dsq = digitize(seq);
        let l = (dsq.len() - 2) as i32;
        for pass in [
            PLI_PASS_5P_ONLY_FORCE,
            PLI_PASS_3P_ONLY_FORCE,
            PLI_PASS_5P_AND_3P_FORCE,
        ] {
            let (_hc, cyk_vsc, _mc, _bc) = ref_tr_cyk_scan(
                &cm, &trp, &dsq, 1, l, -1.0e30, false, SMX_NOQDB, pass, false,
            );
            let (_hi, ins_vsc, _mi, _bi) = ref_itr_inside_scan(
                &cm, &trp, &dsq, 1, l, -1.0e30, false, SMX_NOQDB, pass, false,
            );
            eprintln!("pass {pass}: cyk_vsc={cyk_vsc:.4} inside_vsc={ins_vsc:.4}");
            assert!(
                ins_vsc >= cyk_vsc - 1e-2,
                "pass {pass}: integer Inside score {ins_vsc} < CYK score {cyk_vsc}"
            );
            assert!(ins_vsc.is_finite() && ins_vsc > 0.0, "Inside score must be finite/positive");
        }
    }

    /// Build fully-permissive HMM bands over the whole subseq [i0..j0]: every state
    /// spans j in [i0,j0] and d in [hdmin..min(jp+1,W)] with all 4 marginal planes
    /// valid. hdmin mirrors the non-HB scanner's dn: emitters start at d=1 (they emit
    /// >=1 residue; d=0 is meaningless and would read a sentinel), non-emitters (D/S/B/E)
    /// start at d=0 to permit empty subtrees exactly as the non-HB k=0/k=d cases do.
    fn wide_bands(cm: &CM, i0: i32, j0: i32) -> CP9Bands {
        use crate::constants::E_ST as EE;
        let m = cm.m as usize;
        let w = j0 - i0 + 1;
        let nj = w as usize;
        let mut hdmin = vec![Vec::new(); m];
        let mut hdmax = vec![Vec::new(); m];
        for v in 0..m {
            let stt = cm.sttype[v] as i32;
            let is_e = stt == EE;
            let is_emit = stt == ML_ST || stt == MR_ST || stt == IL_ST || stt == IR_ST || stt == MP_ST;
            let mut hmn = vec![0i32; nj];
            let mut hmx = vec![0i32; nj];
            for jp in 0..nj {
                let dmax = ((jp + 1) as i32).min(w);
                if is_e {
                    hmn[jp] = 0;
                    hmx[jp] = 0;
                } else if is_emit {
                    hmn[jp] = 1;
                    hmx[jp] = dmax.max(1);
                } else {
                    hmn[jp] = 0;
                    hmx[jp] = dmax;
                }
            }
            hdmin[v] = hmn;
            hdmax[v] = hmx;
        }
        CP9Bands {
            hmm_m: 0,
            cm_m: m as i32,
            pn_min_m: Vec::new(), pn_max_m: Vec::new(),
            pn_min_i: Vec::new(), pn_max_i: Vec::new(),
            pn_min_d: Vec::new(), pn_max_d: Vec::new(),
            imin: Vec::new(), imax: Vec::new(),
            jmin: vec![i0; m], jmax: vec![j0; m],
            hdmin, hdmax,
            sp1: -1, sp2: -1, ep1: -1, ep2: -1,
            rmarg_imin: -1, rmarg_imax: -2, lmarg_jmin: -1, lmarg_jmax: -2,
            thresh1: 0.01, thresh2: 0.98,
            jvalid: vec![true; m], lvalid: vec![true; m],
            rvalid: vec![true; m], tvalid: vec![true; m],
        }
    }

    /// Cross-check: with fully-permissive (wide-open) HMM bands, the HB truncated CYK
    /// scan `tr_cyk_scan_hb` must reproduce the byte-verified non-HB `ref_tr_cyk_scan`
    /// vsc_root/vmode exactly (both compute the same truncated-CYK maximum over the
    /// whole space; the only difference is the deck iteration strategy).
    #[test]
    fn tr_cyk_scan_hb_wideband_equals_nonhb() {
        let (cm, trp) = setup();
        let seq = "gCcggcAUAGcgcAgUGGuAgcgCgccagccUgucAagcuggAGgUCCgggGUUCGAUUCcccGUgccgGca";
        let dsq = digitize(seq);
        let l = (dsq.len() - 2) as i32;
        let cp9b = wide_bands(&cm, 1, l);
        for pass in [
            PLI_PASS_5P_ONLY_FORCE,
            PLI_PASS_3P_ONLY_FORCE,
            PLI_PASS_5P_AND_3P_FORCE,
        ] {
            let (_h, vsc_nb, vmode_nb, _b) =
                ref_tr_cyk_scan(&cm, &trp, &dsq, 1, l, -1.0e30, false, SMX_NOQDB, pass, false);
            let (_hh, vsc_hb, vmode_hb, _ei, _ej) =
                tr_cyk_scan_hb(&cm, &trp, &cp9b, &dsq, 1, l, -1.0e30, false, pass, -1.0e30, false);
            eprintln!(
                "pass {pass}: non-HB vsc={vsc_nb:.6} mode={vmode_nb}  HB vsc={vsc_hb:.6} mode={vmode_hb}"
            );
            assert!(
                (vsc_nb - vsc_hb).abs() < 1e-4,
                "pass {pass}: HB wideband vsc {vsc_hb} != non-HB vsc {vsc_nb}"
            );
            assert_eq!(vmode_nb, vmode_hb, "pass {pass}: marginal mode mismatch");
        }
    }

    /// Stage-3 cross-check: with fully-permissive (wide-open) HMM bands, the HB
    /// truncated Inside scan `ftr_inside_scan_hb` must reproduce the byte-verified
    /// non-HB `ref_itr_inside_scan` vsc_root/vmode. The non-HB reference is the
    /// INTEGER Inside (ILogsum); the HB scan is the FLOAT Inside (FLogsum). They
    /// compute the same log-sum-exp maximum but via different-precision semirings,
    /// so we compare within the int-vs-float quantization tolerance (~1e-2 bits),
    /// and require the marginal mode to match exactly.
    #[test]
    fn ftr_inside_scan_hb_wideband_equals_nonhb() {
        let (cm, trp) = setup();
        let seq = "gCcggcAUAGcgcAgUGGuAgcgCgccagccUgucAagcuggAGgUCCgggGUUCGAUUCcccGUgccgGca";
        let dsq = digitize(seq);
        let l = (dsq.len() - 2) as i32;
        let cp9b = wide_bands(&cm, 1, l);
        for pass in [
            PLI_PASS_5P_ONLY_FORCE,
            PLI_PASS_3P_ONLY_FORCE,
            PLI_PASS_5P_AND_3P_FORCE,
        ] {
            let (_h, vsc_nb, vmode_nb, _b) =
                ref_itr_inside_scan(&cm, &trp, &dsq, 1, l, -1.0e30, false, SMX_NOQDB, pass, false);
            let (_hh, vsc_hb, vmode_hb, _ei, _ej) =
                ftr_inside_scan_hb(&cm, &trp, &cp9b, &dsq, 1, l, -1.0e30, false, pass, -1.0e30, false);
            eprintln!(
                "pass {pass}: non-HB(int) vsc={vsc_nb:.6} mode={vmode_nb}  HB(float) vsc={vsc_hb:.6} mode={vmode_hb}"
            );
            assert!(
                (vsc_nb - vsc_hb).abs() < 2e-2,
                "pass {pass}: HB wideband Inside vsc {vsc_hb} != non-HB Inside vsc {vsc_nb}"
            );
            assert_eq!(vmode_nb, vmode_hb, "pass {pass}: marginal mode mismatch");
        }
    }
}
