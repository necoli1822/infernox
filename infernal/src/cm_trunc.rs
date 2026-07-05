//! cm_trunc — faithful port of C Infernal 1.1.5 non-banded truncated CYK
//! alignment (TrCYK) for the `-g --toponly` global search path.
//!
//! Ports (C fn -> Rust fn):
//!   * `cm_tr_penalties_Create` (cm_trunc.c:51, GLOBAL arrays only, since `-g`)
//!       -> [`TrPenalties::new`]
//!   * `cm_tr_penalties_IdxForPass` (cm_trunc.c:464) -> [`pty_idx_for_pass`]
//!   * `cm_TrCYKInsideAlign` (cm_dpalign_trunc.c:1274) + `cm_tr_alignT`
//!       (cm_dpalign_trunc.c:178) -> [`tr_cyk_align`] (fill + traceback, marginal
//!       modes J/L/R/T, non-banded, global: no local begins/ends/EL)
//!   * `ParsetreeToCMBounds` (cm_parsetree.c:2611) -> [`parsetree_to_cm_bounds`]
//!
//! Only the CYK path is ported (do_optacc = do_post = FALSE). Global config only
//! (`-g`): `cm.endsc[v]` is IMPOSSIBLE everywhere and `CMH_LOCAL_END`/`_BEGIN` are
//! off, so the EL deck and local-begin bookkeeping are omitted; truncated begins
//! use the GLOBAL penalty arrays `g_ptyAA`.

use crate::cm::CM;
use crate::constants::{
    B_ST, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST,
    MATP_ND, MATL_ND, MATR_ND, BIF_ND, BEGR_ND, ROOT_ND, END_ND,
    MATP_MP, MATL_ML, MATR_MR, BIF_B, MATP_IL, END_E,
};
use crate::cp9_faithful::EmitMap;
use crate::legacy::parsetree::Parsetree;

const IMPOSSIBLE: f32 = -1.0e36;
const K: usize = 4;

// Truncation marginal modes (infernal.h).
pub const TRMODE_T: i8 = 0;
pub const TRMODE_R: i8 = 1;
pub const TRMODE_L: i8 = 2;
pub const TRMODE_J: i8 = 3;
pub const TRMODE_UNKNOWN: i8 = 4;
const TRMODE_J_OFFSET: i32 = 0;
const TRMODE_L_OFFSET: i32 = 10;
const TRMODE_R_OFFSET: i32 = 20;

// Shadow-cell sentinels (infernal.h). Stored in i32 shadow decks.
const USED_EL: i32 = 102;
const USED_TRUNC_BEGIN: i32 = 103;
const USED_TRUNC_END: i32 = 104;

// Pipeline pass indices (infernal.h).
pub const PLI_PASS_STD_ANY: i32 = 1;
pub const PLI_PASS_5P_ONLY_FORCE: i32 = 2;
pub const PLI_PASS_3P_ONLY_FORCE: i32 = 3;
pub const PLI_PASS_5P_AND_3P_FORCE: i32 = 4;

// Truncation-penalty array indices (infernal.h).
const TRPENALTY_5P_AND_3P: usize = 0;
const TRPENALTY_5P_ONLY: usize = 1;
const TRPENALTY_3P_ONLY: usize = 2;

#[inline]
fn not_impossible(x: f32) -> bool {
    x > -0.5e36
}

// ---- FLogsum (Infernal logsum.c, bits/log2 lookup table) ----
const LOGSUM_TBL: usize = 23000;
const INTSCALE: f32 = 1000.0;
static FLOGSUM_TBL: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

fn flogsum_table() -> &'static [f32] {
    FLOGSUM_TBL.get_or_init(|| {
        // flogsum_lookup[i] = log2(1 + 2^(-i/INTSCALE))  (logsum.c:150-152)
        (0..LOGSUM_TBL)
            .map(|i| (1.0f64 + 2f64.powf(-(i as f64) / INTSCALE as f64)).log2() as f32)
            .collect()
    })
}

/// C `FLogsum` (logsum.c): log2(2^s1 + 2^s2) via a lookup table, gated at 23 bits.
#[inline]
fn flogsum(s1: f32, s2: f32) -> f32 {
    let (max, min) = if s1 >= s2 { (s1, s2) } else { (s2, s1) };
    let diff = max - min;
    if diff >= 23.0 {
        max
    } else {
        max + flogsum_table()[(diff * INTSCALE) as usize]
    }
}

/// C `cm_tr_penalties_IdxForPass` (cm_trunc.c:464). Returns the g_ptyAA index for a
/// truncated pass, or `None` for the standard pass.
pub fn pty_idx_for_pass(pass_idx: i32) -> Option<usize> {
    match pass_idx {
        PLI_PASS_5P_ONLY_FORCE => Some(TRPENALTY_5P_ONLY),
        PLI_PASS_3P_ONLY_FORCE => Some(TRPENALTY_3P_ONLY),
        PLI_PASS_5P_AND_3P_FORCE => Some(TRPENALTY_5P_AND_3P),
        _ => None,
    }
}

/// C `InsertsGivenNodeIndex` (cm.c). Returns (i1, i2) parent-insert states of node
/// `nd` (−1 if none).
fn inserts_given_node(cm: &CM, nd: i32) -> (i32, i32) {
    if nd < 0 {
        return (-1, -1);
    }
    let v = cm.nodemap[nd as usize];
    match cm.ndtype[nd as usize] as i32 {
        MATP_ND => (v + 4, v + 5),
        MATL_ND => (v + 2, -1),
        MATR_ND => (v + 2, -1),
        BEGR_ND => (v + 1, -1),
        ROOT_ND => (v + 1, v + 2),
        _ => (-1, -1),
    }
}

#[inline]
fn state_is_detached(cm: &CM, v: usize) -> bool {
    (cm.stid[v + 1] as i32) == END_E
}

/// GLOBAL truncated-begin penalty arrays (C `trp->g_ptyAA`), log2, per state.
/// `g[idx][v]` is the truncated-begin penalty for entering state `v` under pass
/// index `idx` (5P_AND_3P / 5P_ONLY / 3P_ONLY). IMPOSSIBLE = disallowed.
pub struct TrPenalties {
    pub g: [Vec<f32>; 3],
}

impl TrPenalties {
    /// Port of `cm_tr_penalties_Create` (cm_trunc.c:51), global arrays only.
    /// `psi` = expected state occupancy (cm_expected_state_occupancy of the same
    /// global-config CM). `emap` = its emit map.
    pub fn new(cm: &CM, emap: &EmitMap, psi: &[f64]) -> Self {
        let m = cm.m as usize;
        let clen = cm.clen;
        let mut g: [Vec<f32>; 3] = [
            vec![IMPOSSIBLE; m],
            vec![IMPOSSIBLE; m],
            vec![IMPOSSIBLE; m],
        ];
        let g_5and3 = 2.0f32 / (clen as f32 * (clen as f32 + 1.0));
        let g_5or3 = 1.0f32 / clen as f32;

        for nd in 0..cm.nodes as usize {
            let ndt = cm.ndtype[nd] as i32;
            let lpos = if ndt == MATP_ND || ndt == MATL_ND {
                emap.lpos[nd]
            } else {
                emap.lpos[nd] + 1
            };
            let rpos = if ndt == MATP_ND || ndt == MATR_ND {
                emap.rpos[nd]
            } else {
                emap.rpos[nd] - 1
            };
            if ndt == MATP_ND || ndt == MATL_ND || ndt == MATR_ND || ndt == BIF_ND {
                let mst = cm.nodemap[nd] as usize;
                let (i1, i2) = inserts_given_node(cm, nd as i32 - 1);
                let mut m_psi = psi[mst];
                if ndt == MATP_ND {
                    m_psi += psi[mst + 1] + psi[mst + 2];
                }
                let i1_psi = if i1 >= 0 { psi[i1 as usize] } else { 0.0 };
                let i2_psi = if i2 >= 0 { psi[i2 as usize] } else { 0.0 };
                let summed = m_psi + i1_psi + i2_psi;

                let set = |g: &mut Vec<f32>, prob: f32| {
                    g[mst] = (m_psi / summed) as f32 * prob;
                    if i1 >= 0 {
                        g[i1 as usize] = (i1_psi / summed) as f32 * prob;
                    }
                    if i2 >= 0 {
                        g[i2 as usize] = (i2_psi / summed) as f32 * prob;
                    }
                };
                set(&mut g[TRPENALTY_5P_AND_3P], g_5and3);
                if rpos == clen {
                    set(&mut g[TRPENALTY_5P_ONLY], g_5or3);
                }
                if lpos == 1 {
                    set(&mut g[TRPENALTY_3P_ONLY], g_5or3);
                }
            }
        }

        // Convert probabilities to log2 penalties for the qualifying states.
        for v in 0..m {
            let stid = cm.stid[v] as i32;
            let stt = cm.sttype[v] as i32;
            let qualifies = stid == MATP_MP
                || stid == MATL_ML
                || stid == MATR_MR
                || stid == BIF_B
                || ((stt == IL_ST || stt == IR_ST) && !state_is_detached(cm, v));
            if !qualifies {
                continue;
            }
            // Rare special case: MATP_IL followed by END keeps IMPOSSIBLE.
            if stid == MATP_IL && (cm.ndtype[(cm.ndidx[v] + 1) as usize] as i32) == END_ND {
                continue;
            }
            for idx in 0..3 {
                if not_impossible(g[idx][v]) {
                    g[idx][v] = (g[idx][v] as f64).log2() as f32;
                }
            }
        }

        TrPenalties { g }
    }
}

// ---- emission helpers (canonical fast path + degenerate average fallback) ----
#[inline]
fn pair_sc(esc: &[f32], di: u8, dj: u8) -> f32 {
    if (di as usize) < K && (dj as usize) < K {
        esc[di as usize * K + dj as usize]
    } else {
        let mut s = 0.0f32;
        for a in 0..K {
            for c in 0..K {
                s += esc[a * K + c];
            }
        }
        s / (K * K) as f32
    }
}
#[inline]
fn sing_sc(esc: &[f32], di: u8) -> f32 {
    if (di as usize) < K {
        esc[di as usize]
    } else {
        let mut s = 0.0f32;
        for a in 0..K {
            s += esc[a];
        }
        s / K as f32
    }
}
#[inline]
fn marg_sc(m: &[f32; 4], di: u8) -> f32 {
    if (di as usize) < K {
        m[di as usize]
    } else {
        (m[0] + m[1] + m[2] + m[3]) / K as f32
    }
}

/// Marginal left/right emission log-odds for each MP state (C `cm->lmesc`/`rmesc`),
/// derived from `cm.e` and `cm.null`. `[f32;4]` per state; canonical residues only.
fn build_marginal_emissions(cm: &CM) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let m = cm.m as usize;
    let mut lm = vec![[0.0f32; 4]; m];
    let mut rm = vec![[0.0f32; 4]; m];
    for v in 0..m {
        if cm.sttype[v] as i32 != MP_ST {
            continue;
        }
        for x in 0..K {
            let mut ls = 0.0f32;
            let mut rs = 0.0f32;
            for y in 0..K {
                ls += cm.e[v][x * K + y];
                rs += cm.e[v][y * K + x];
            }
            lm[v][x] = (ls / cm.null[x]).log2();
            rm[v][x] = (rs / cm.null[x]).log2();
        }
    }
    (lm, rm)
}

/// Non-banded truncated CYK alignment of `dsq[1..=lp]` to the GLOBAL-config CM,
/// for pipeline pass `pass_idx` (one of the FORCE truncated passes). Fills the
/// J/L/R/T marginal matrices (preset_mode = UNKNOWN, best mode chosen by score),
/// traces back and returns (parsetree-with-modes, score, mode).
///
/// Port of `cm_TrCYKInsideAlign` + `cm_tr_alignT` (CYK path). `dsq[0]` is unused.
pub fn tr_cyk_align(
    cm: &CM,
    trp: &TrPenalties,
    lm: &[[f32; 4]],
    rm: &[[f32; 4]],
    dsq: &[u8],
    lp: i32,
    pass_idx: i32,
) -> (Parsetree, f32, i8) {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let ncell = stride * stride;
    let pty = trp.g[pty_idx_for_pass(pass_idx).expect("truncated pass")].as_slice();

    // per-state flat decks, index j*stride+d
    let mut jalpha: Vec<Vec<f32>> = vec![Vec::new(); m];
    let mut lalpha: Vec<Vec<f32>> = vec![Vec::new(); m];
    let mut ralpha: Vec<Vec<f32>> = vec![Vec::new(); m];
    let mut talpha: Vec<Vec<f32>> = vec![Vec::new(); m]; // B states only
    let mut jyshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut lyshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut ryshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut jkshad: Vec<Vec<i32>> = vec![Vec::new(); m]; // B states
    let mut lkshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut rkshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut tkshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut lkmode: Vec<Vec<i8>> = vec![Vec::new(); m];
    let mut rkmode: Vec<Vec<i8>> = vec![Vec::new(); m];

    // ROOT [0][L][L] scalars + best entry states per mode.
    let mut j0 = IMPOSSIBLE;
    let mut l0 = IMPOSSIBLE;
    let mut r0 = IMPOSSIBLE;
    let mut t0 = IMPOSSIBLE;
    let (mut jb, mut lb, mut rb, mut tb) = (0i32, 0i32, 0i32, 0i32);

    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];

        // local decks (global: init all IMPOSSIBLE; no endsc/EL re-init)
        let mut ja = vec![IMPOSSIBLE; ncell];
        let mut la = vec![IMPOSSIBLE; ncell];
        let mut ra = vec![IMPOSSIBLE; ncell];
        let mut ta = vec![IMPOSSIBLE; ncell];
        let mut jy = vec![USED_EL; ncell];
        let mut ly = vec![USED_EL; ncell];
        let mut ry = vec![USED_EL; ncell];
        let mut jk = vec![0i32; ncell];
        let mut lk = vec![0i32; ncell];
        let mut rk = vec![0i32; ncell];
        let mut tk = vec![0i32; ncell];
        let mut lkm = vec![TRMODE_J; ncell];
        let mut rkm = vec![TRMODE_J; ncell];

        if stt == E_ST {
            for j in 0..=w {
                ja[j * stride] = 0.0;
                la[j * stride] = 0.0;
                ra[j * stride] = 0.0;
            }
        } else if stt == IL_ST || stt == ML_ST {
            if !state_is_detached(cm, v) {
                let ryoffset0 = if stt == IL_ST { 1 } else { 0 };
                for j in 0..=w {
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let dm1 = j * stride + (d - 1);
                        let i = j - d + 1;
                        for yo in 0..cnum {
                            let y = cfirst + yo;
                            let (jc, lc) = if y == v {
                                (ja[dm1], la[dm1])
                            } else {
                                (jalpha[y][dm1], lalpha[y][dm1])
                            };
                            let sc = jc + tsc_v[yo];
                            if sc > ja[idx] {
                                ja[idx] = sc;
                                jy[idx] = yo as i32 + TRMODE_J_OFFSET;
                            }
                            let sc = lc + tsc_v[yo];
                            if sc > la[idx] {
                                la[idx] = sc;
                                ly[idx] = yo as i32 + TRMODE_L_OFFSET;
                            }
                        }
                        let e = sing_sc(esc_v, dsq[i]);
                        ja[idx] += e;
                        if ja[idx] < IMPOSSIBLE {
                            ja[idx] = IMPOSSIBLE;
                        }
                        if d >= 2 {
                            la[idx] += e;
                        } else {
                            la[idx] = e;
                            ly[idx] = USED_TRUNC_END;
                        }
                        if la[idx] < IMPOSSIBLE {
                            la[idx] = IMPOSSIBLE;
                        }
                        // R matrix (uses 'd', reads [j][d])
                        let rd = j * stride + d;
                        for yo in ryoffset0..cnum {
                            let y = cfirst + yo;
                            let (jc, rc) = if y == v {
                                (ja[rd], ra[rd])
                            } else {
                                (jalpha[y][rd], ralpha[y][rd])
                            };
                            let sc = jc + tsc_v[yo];
                            if sc > ra[idx] {
                                ra[idx] = sc;
                                ry[idx] = yo as i32 + TRMODE_J_OFFSET;
                            }
                            let sc = rc + tsc_v[yo];
                            if sc > ra[idx] {
                                ra[idx] = sc;
                                ry[idx] = yo as i32 + TRMODE_R_OFFSET;
                            }
                        }
                        if ra[idx] < IMPOSSIBLE {
                            ra[idx] = IMPOSSIBLE;
                        }
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            if !state_is_detached(cm, v) {
                let lyoffset0 = if stt == IR_ST { 1 } else { 0 };
                for j in 1..=w {
                    let jm1 = (j - 1) * stride;
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let dm1 = jm1 + (d - 1);
                        for yo in 0..cnum {
                            let y = cfirst + yo;
                            let (jc, rc) = if y == v {
                                (ja[dm1], ra[dm1])
                            } else {
                                (jalpha[y][dm1], ralpha[y][dm1])
                            };
                            let sc = jc + tsc_v[yo];
                            if sc > ja[idx] {
                                ja[idx] = sc;
                                jy[idx] = yo as i32 + TRMODE_J_OFFSET;
                            }
                            let sc = rc + tsc_v[yo];
                            if sc > ra[idx] {
                                ra[idx] = sc;
                                ry[idx] = yo as i32 + TRMODE_R_OFFSET;
                            }
                        }
                        let e = sing_sc(esc_v, dsq[j]);
                        ja[idx] += e;
                        if ja[idx] < IMPOSSIBLE {
                            ja[idx] = IMPOSSIBLE;
                        }
                        if d >= 2 {
                            ra[idx] += e;
                        } else {
                            ra[idx] = e;
                            ry[idx] = USED_TRUNC_END;
                        }
                        if ra[idx] < IMPOSSIBLE {
                            ra[idx] = IMPOSSIBLE;
                        }
                        // L matrix (uses 'j','d', reads [j][d])
                        let ld = j * stride + d;
                        for yo in lyoffset0..cnum {
                            let y = cfirst + yo;
                            let (jc, lc) = if y == v {
                                (ja[ld], la[ld])
                            } else {
                                (jalpha[y][ld], lalpha[y][ld])
                            };
                            let sc = jc + tsc_v[yo];
                            if sc > la[idx] {
                                la[idx] = sc;
                                ly[idx] = yo as i32 + TRMODE_J_OFFSET;
                            }
                            let sc = lc + tsc_v[yo];
                            if sc > la[idx] {
                                la[idx] = sc;
                                ly[idx] = yo as i32 + TRMODE_L_OFFSET;
                            }
                        }
                        if la[idx] < IMPOSSIBLE {
                            la[idx] = IMPOSSIBLE;
                        }
                    }
                }
            }
        } else if stt == MP_ST {
            // transition loop (no self-transit): for y { for j>=1 { J/L/R } }
            for yo in 0..cnum {
                let y = cfirst + yo;
                let tsc = tsc_v[yo];
                for j in 1..=w {
                    let jm1 = (j - 1) * stride;
                    // J: d from 2
                    for d in 2..=j {
                        let sc = jalpha[y][jm1 + (d - 2)] + tsc;
                        let idx = j * stride + d;
                        if sc > ja[idx] {
                            ja[idx] = sc;
                            jy[idx] = yo as i32 + TRMODE_J_OFFSET;
                        }
                    }
                    // L: d from 1, uses [j][d-1]
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let src = j * stride + (d - 1);
                        let sc = jalpha[y][src] + tsc;
                        if sc > la[idx] {
                            la[idx] = sc;
                            ly[idx] = yo as i32 + TRMODE_J_OFFSET;
                        }
                        let sc = lalpha[y][src] + tsc;
                        if sc > la[idx] {
                            la[idx] = sc;
                            ly[idx] = yo as i32 + TRMODE_L_OFFSET;
                        }
                    }
                    // R: d from 1, uses [j-1][d-1]
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let src = jm1 + (d - 1);
                        let sc = jalpha[y][src] + tsc;
                        if sc > ra[idx] {
                            ra[idx] = sc;
                            ry[idx] = yo as i32 + TRMODE_J_OFFSET;
                        }
                        let sc = ralpha[y][src] + tsc;
                        if sc > ra[idx] {
                            ra[idx] = sc;
                            ry[idx] = yo as i32 + TRMODE_R_OFFSET;
                        }
                    }
                }
            }
            // emission loop
            let lmv = &lm[v];
            let rmv = &rm[v];
            for j in 0..=w {
                let idx1 = j * stride + 1;
                ja[idx1] = IMPOSSIBLE;
                if j >= 1 {
                    la[idx1] = marg_sc(lmv, dsq[j]);
                    ly[idx1] = USED_TRUNC_END;
                    ra[idx1] = marg_sc(rmv, dsq[j]);
                    ry[idx1] = USED_TRUNC_END;
                }
                let mut i = if j >= 1 { j - 1 } else { 0 };
                for d in 2..=j {
                    let idx = j * stride + d;
                    ja[idx] += pair_sc(esc_v, dsq[i], dsq[j]);
                    la[idx] += marg_sc(lmv, dsq[i]);
                    ra[idx] += marg_sc(rmv, dsq[j]);
                    i = i.wrapping_sub(1);
                }
            }
            // clamp
            for j in 0..=w {
                for d in 1..=j {
                    let idx = j * stride + d;
                    if ja[idx] < IMPOSSIBLE {
                        ja[idx] = IMPOSSIBLE;
                    }
                    if la[idx] < IMPOSSIBLE {
                        la[idx] = IMPOSSIBLE;
                    }
                    if ra[idx] < IMPOSSIBLE {
                        ra[idx] = IMPOSSIBLE;
                    }
                }
            }
        } else if stt != B_ST {
            // D or S states (no self-transit, no emission)
            for yo in 0..cnum {
                let y = cfirst + yo;
                let tsc = tsc_v[yo];
                for j in 0..=w {
                    let jrow = j * stride;
                    for d in 0..=j {
                        let idx = jrow + d;
                        let src = jrow + d; // sdr=0, sd=0
                        let sc = jalpha[y][src] + tsc;
                        if sc > ja[idx] {
                            ja[idx] = sc;
                            jy[idx] = yo as i32 + TRMODE_J_OFFSET;
                        }
                        let sc = lalpha[y][src] + tsc;
                        if sc > la[idx] {
                            la[idx] = sc;
                            ly[idx] = yo as i32 + TRMODE_L_OFFSET;
                        }
                        let sc = ralpha[y][src] + tsc;
                        if sc > ra[idx] {
                            ra[idx] = sc;
                            ry[idx] = yo as i32 + TRMODE_R_OFFSET;
                        }
                    }
                    // d==0 L/R forced IMPOSSIBLE; S states get USED_TRUNC_END shadow
                    la[jrow] = IMPOSSIBLE;
                    ra[jrow] = IMPOSSIBLE;
                    if stt == S_ST {
                        ly[jrow] = USED_TRUNC_END;
                        ry[jrow] = USED_TRUNC_END;
                    }
                }
            }
        } else {
            // B state
            let y = cfirst; // left  (BEGL_S)
            let z = cm.cnum[v] as usize; // right (BEGR_S)
            for j in 0..=w {
                let jrow = j * stride;
                for d in 0..=j {
                    let idx = jrow + d;
                    for k in 0..=d {
                        let ly_src = (j - k) * stride + (d - k);
                        let z_src = jrow + k;
                        let sc = jalpha[y][ly_src] + jalpha[z][z_src];
                        if sc > ja[idx] {
                            ja[idx] = sc;
                            jk[idx] = k as i32;
                        }
                        let sc = jalpha[y][ly_src] + lalpha[z][z_src];
                        if sc > la[idx] {
                            la[idx] = sc;
                            lk[idx] = k as i32;
                            lkm[idx] = TRMODE_J;
                        }
                        let sc = ralpha[y][ly_src] + jalpha[z][z_src];
                        if sc > ra[idx] {
                            ra[idx] = sc;
                            rk[idx] = k as i32;
                            rkm[idx] = TRMODE_J;
                        }
                    }
                    // T matrix: k in 1..d
                    for k in 1..d {
                        let ly_src = (j - k) * stride + (d - k);
                        let z_src = jrow + k;
                        let sc = ralpha[y][ly_src] + lalpha[z][z_src];
                        if sc > ta[idx] {
                            ta[idx] = sc;
                            tk[idx] = k as i32;
                        }
                    }
                    // special case 1: k==0 (full seq on left)
                    let sc = jalpha[y][idx];
                    if sc > la[idx] {
                        la[idx] = sc;
                        lk[idx] = 0;
                        lkm[idx] = TRMODE_J;
                    }
                    let sc = lalpha[y][idx];
                    if sc > la[idx] {
                        la[idx] = sc;
                        lk[idx] = 0;
                        lkm[idx] = TRMODE_L;
                    }
                    // special case 2: k==d (full seq on right)
                    let sc = jalpha[z][idx];
                    if sc > ra[idx] {
                        ra[idx] = sc;
                        rk[idx] = d as i32;
                        rkm[idx] = TRMODE_J;
                    }
                    let sc = ralpha[z][idx];
                    if sc > ra[idx] {
                        ra[idx] = sc;
                        rk[idx] = d as i32;
                        rkm[idx] = TRMODE_R;
                    }
                }
            }
        }

        // ROOT truncated-begin update.
        let trpenalty = pty[v];
        if not_impossible(trpenalty) {
            let root = w * stride + w;
            let sc = ja[root] + trpenalty;
            if sc > j0 {
                j0 = sc;
                jb = v as i32;
            }
            let sc = la[root] + trpenalty;
            if sc > l0 {
                l0 = sc;
                lb = v as i32;
            }
            let sc = ra[root] + trpenalty;
            if sc > r0 {
                r0 = sc;
                rb = v as i32;
            }
            if stt == B_ST {
                let sc = ta[root] + trpenalty;
                if sc > t0 {
                    t0 = sc;
                    tb = v as i32;
                }
            }
        }

        jalpha[v] = ja;
        lalpha[v] = la;
        ralpha[v] = ra;
        if stt == B_ST {
            talpha[v] = ta;
        }
        jyshad[v] = jy;
        lyshad[v] = ly;
        ryshad[v] = ry;
        if stt == B_ST {
            jkshad[v] = jk;
            lkshad[v] = lk;
            rkshad[v] = rk;
            tkshad[v] = tk;
            lkmode[v] = lkm;
            rkmode[v] = rkm;
        }
    }

    // choose optimal mode (preset_mode = UNKNOWN)
    let mut sc = j0;
    let mut mode = TRMODE_J;
    let mut b = jb;
    if l0 > sc {
        sc = l0;
        mode = TRMODE_L;
        b = lb;
    }
    if r0 > sc {
        sc = r0;
        mode = TRMODE_R;
        b = rb;
    }
    if t0 > sc {
        sc = t0;
        mode = TRMODE_T;
        b = tb;
    }

    // ---- traceback (cm_tr_alignT, CYK path) ----
    let opt_mode = mode;
    let mut tr = Parsetree::new(w + 4);
    tr.add_node_mode(1, lp, 0, -1, -1, -1, mode); // root
    // pda_i frames pushed as (bifparent, k, j) via 3 i32 pushes; pda_c mode
    let mut pda_i: Vec<i32> = Vec::new();
    let mut pda_c: Vec<i8> = Vec::new();
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = lp;
    let mut d: i32 = lp;

    loop {
        let vu = v as usize;
        // EL: never in global; treat like end swing
        if v == cm.m || (v != 0 && cm.sttype[vu] as i32 == E_ST) {
            match pda_i.pop() {
                None => break,
                Some(bifparent) => {
                    d = pda_i.pop().unwrap();
                    j = pda_i.pop().unwrap();
                    mode = pda_c.pop().unwrap();
                    let bstate = tr.state[bifparent as usize] as usize;
                    let yr = cm.cnum[bstate];
                    i = j - d + 1;
                    let idx = tr.add_node_mode(i, j, yr, -1, -1, bifparent, mode);
                    tr.nxtr[bifparent as usize] = idx;
                    v = yr;
                    continue;
                }
            }
        }
        let stt = cm.sttype[vu] as i32;
        let didx = (j as usize) * stride + d as usize;
        if stt == B_ST {
            let k = match mode {
                TRMODE_J => jkshad[vu][didx],
                TRMODE_L => lkshad[vu][didx],
                TRMODE_R => rkshad[vu][didx],
                _ => tkshad[vu][didx], // TRMODE_T
            };
            let prvmode = mode;
            // right child mode
            let rmode = match prvmode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => TRMODE_L,
                TRMODE_R => rkmode[vu][didx],
                _ => TRMODE_L, // T
            };
            let bpar = tr.n - 1;
            pda_c.push(rmode);
            pda_i.push(j);
            pda_i.push(k);
            pda_i.push(bpar);
            // left child mode
            let lmode = match prvmode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => lkmode[vu][didx],
                TRMODE_R => TRMODE_R,
                _ => TRMODE_R, // T
            };
            j -= k;
            d -= k;
            i = j - d + 1;
            let yl = cm.cfirst[vu];
            let idx = tr.add_node_mode(i, j, yl, -1, -1, bpar, lmode);
            tr.nxtl[bpar as usize] = idx;
            v = yl;
            mode = lmode;
            continue;
        }

        // non-B, non-E, non-EL
        let yoffset_raw = if v == 0 {
            USED_TRUNC_BEGIN
        } else {
            match mode {
                TRMODE_J => jyshad[vu][didx],
                TRMODE_L => lyshad[vu][didx],
                TRMODE_R => ryshad[vu][didx],
                _ => USED_TRUNC_BEGIN, // T at v==0 handled above; shouldn't reach for others
            }
        };
        let mut nxtmode = mode;
        let yoffset;
        if yoffset_raw == USED_TRUNC_BEGIN
            || yoffset_raw == USED_TRUNC_END
            || yoffset_raw == USED_EL
        {
            yoffset = yoffset_raw;
        } else if yoffset_raw >= TRMODE_R_OFFSET {
            nxtmode = TRMODE_R;
            yoffset = yoffset_raw - TRMODE_R_OFFSET;
        } else if yoffset_raw >= TRMODE_L_OFFSET {
            nxtmode = TRMODE_L;
            yoffset = yoffset_raw - TRMODE_L_OFFSET;
        } else {
            nxtmode = TRMODE_J;
            yoffset = yoffset_raw - TRMODE_J_OFFSET;
        }

        // emit position updates (mode-dependent)
        match stt {
            x if x == MP_ST => {
                if mode == TRMODE_J {
                    i += 1;
                }
                if mode == TRMODE_L && d > 0 {
                    i += 1;
                }
                if mode == TRMODE_J {
                    j -= 1;
                }
                if mode == TRMODE_R && d > 0 {
                    j -= 1;
                }
            }
            x if x == ML_ST || x == IL_ST => {
                if mode == TRMODE_J {
                    i += 1;
                }
                if mode == TRMODE_L && d > 0 {
                    i += 1;
                }
            }
            x if x == MR_ST || x == IR_ST => {
                if mode == TRMODE_J {
                    j -= 1;
                }
                if mode == TRMODE_R && d > 0 {
                    j -= 1;
                }
            }
            _ => {} // D, S
        }
        d = j - i + 1;

        if yoffset == USED_EL || yoffset == USED_TRUNC_END {
            v = cm.m; // swing to end
        } else if yoffset == USED_TRUNC_BEGIN {
            let idx = tr.add_node_mode(i, j, b, -1, -1, tr.n - 1, mode);
            tr.nxtl[(tr.n - 2) as usize] = idx;
            v = b;
        } else {
            mode = nxtmode;
            let yy = cm.cfirst[vu] + yoffset;
            let idx = tr.add_node_mode(i, j, yy, -1, -1, tr.n - 1, mode);
            tr.nxtl[(tr.n - 2) as usize] = idx;
            v = yy;
        }
    }

    (tr, sc, opt_mode)
}

/// Standard (non-truncated) global Inside score of `dsq[1..=lp]` (C `cm_InsideAlign`,
/// global config). Log-sum (FLogsum) version of [`crate::cm_alidisplay::cyk_align_global`]'s
/// fill; returns `alpha[0][L][L]`. Used to rank the STD pass against the truncated
/// passes with the SAME objective (Inside) that C's default pipeline uses.
pub fn inside_score_global(cm: &CM, dsq: &[u8], lp: i32) -> f32 {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let ncell = stride * stride;
    let mut alpha: Vec<Vec<f32>> = vec![Vec::new(); m];

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        if stt == E_ST {
            let mut a = vec![IMPOSSIBLE; ncell];
            for j in 0..=w {
                a[j * stride] = 0.0;
            }
            alpha[v] = a;
            continue;
        }
        let mut a = vec![IMPOSSIBLE; ncell];
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];

        if stt == D_ST || stt == S_ST {
            for j in 0..=w {
                for d in 0..=j {
                    let idx = j * stride + d;
                    let mut acc = IMPOSSIBLE;
                    for yo in 0..cnum {
                        acc = flogsum(acc, alpha[cfirst + yo][idx] + tsc_v[yo]);
                    }
                    a[idx] = acc.max(IMPOSSIBLE);
                }
            }
        } else if stt == B_ST {
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            for j in 0..=w {
                for d in 0..=j {
                    let idx = j * stride + d;
                    let mut acc = IMPOSSIBLE;
                    for k in 0..=d {
                        acc = flogsum(acc, alpha[y][(j - k) * stride + (d - k)] + alpha[z][j * stride + k]);
                    }
                    a[idx] = acc.max(IMPOSSIBLE);
                }
            }
        } else if stt == MP_ST {
            for j in 0..=w {
                for d in 2..=j {
                    let idx = j * stride + d;
                    let mut acc = IMPOSSIBLE;
                    for yo in 0..cnum {
                        acc = flogsum(acc, alpha[cfirst + yo][(j - 1) * stride + (d - 2)] + tsc_v[yo]);
                    }
                    let i = j - d + 1;
                    acc += pair_sc(esc_v, dsq[i], dsq[j]);
                    a[idx] = acc.max(IMPOSSIBLE);
                }
            }
        } else if stt == IL_ST || stt == ML_ST {
            for j in 0..=w {
                for d in 1..=j {
                    let idx = j * stride + d;
                    let cidx = j * stride + (d - 1);
                    let mut acc = IMPOSSIBLE;
                    for yo in 0..cnum {
                        let child = cfirst + yo;
                        let cell = if child == v { a[cidx] } else { alpha[child][cidx] };
                        acc = flogsum(acc, cell + tsc_v[yo]);
                    }
                    let i = j - d + 1;
                    acc += sing_sc(esc_v, dsq[i]);
                    a[idx] = acc.max(IMPOSSIBLE);
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            for j in 0..=w {
                for d in 1..=j {
                    let idx = j * stride + d;
                    let cidx = (j - 1) * stride + (d - 1);
                    let mut acc = IMPOSSIBLE;
                    for yo in 0..cnum {
                        let child = cfirst + yo;
                        let cell = if child == v { a[cidx] } else { alpha[child][cidx] };
                        acc = flogsum(acc, cell + tsc_v[yo]);
                    }
                    acc += sing_sc(esc_v, dsq[j]);
                    a[idx] = acc.max(IMPOSSIBLE);
                }
            }
        }
        alpha[v] = a;
    }

    alpha[0][w * stride + w]
}

/// Truncated Inside score of `dsq[1..=lp]` for pipeline pass `pass_idx` (C
/// `cm_TrInsideAlign`, CYK path replaced by log-sum). Returns the max over marginal
/// modes J/L/R/T of the mode's Inside score at `alpha[0][L][L]` (C: "we don't sum
/// over different marginal modes, we pick the highest scoring one"). No shadow /
/// traceback; used only to rank passes. Global config (no EL / local begins).
pub fn tr_inside_score(
    cm: &CM,
    trp: &TrPenalties,
    lm: &[[f32; 4]],
    rm: &[[f32; 4]],
    dsq: &[u8],
    lp: i32,
    pass_idx: i32,
) -> f32 {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let ncell = stride * stride;
    let pty = trp.g[pty_idx_for_pass(pass_idx).expect("truncated pass")].as_slice();

    let mut jalpha: Vec<Vec<f32>> = vec![Vec::new(); m];
    let mut lalpha: Vec<Vec<f32>> = vec![Vec::new(); m];
    let mut ralpha: Vec<Vec<f32>> = vec![Vec::new(); m];
    let mut talpha: Vec<Vec<f32>> = vec![Vec::new(); m];

    let mut j0 = IMPOSSIBLE;
    let mut l0 = IMPOSSIBLE;
    let mut r0 = IMPOSSIBLE;
    let mut t0 = IMPOSSIBLE;

    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];

        let mut ja = vec![IMPOSSIBLE; ncell];
        let mut la = vec![IMPOSSIBLE; ncell];
        let mut ra = vec![IMPOSSIBLE; ncell];
        let mut ta = vec![IMPOSSIBLE; ncell];

        if stt == E_ST {
            for j in 0..=w {
                ja[j * stride] = 0.0;
                la[j * stride] = 0.0;
                ra[j * stride] = 0.0;
            }
        } else if stt == IL_ST || stt == ML_ST {
            if !state_is_detached(cm, v) {
                let ryoffset0 = if stt == IL_ST { 1 } else { 0 };
                for j in 0..=w {
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let dm1 = j * stride + (d - 1);
                        let i = j - d + 1;
                        for yo in 0..cnum {
                            let y = cfirst + yo;
                            let (jc, lc) = if y == v {
                                (ja[dm1], la[dm1])
                            } else {
                                (jalpha[y][dm1], lalpha[y][dm1])
                            };
                            ja[idx] = flogsum(ja[idx], jc + tsc_v[yo]);
                            la[idx] = flogsum(la[idx], lc + tsc_v[yo]);
                        }
                        let e = sing_sc(esc_v, dsq[i]);
                        ja[idx] = (ja[idx] + e).max(IMPOSSIBLE);
                        la[idx] = if d >= 2 { la[idx] + e } else { e };
                        la[idx] = la[idx].max(IMPOSSIBLE);
                        let rd = j * stride + d;
                        for yo in ryoffset0..cnum {
                            let y = cfirst + yo;
                            let (jc, rc) = if y == v {
                                (ja[rd], ra[rd])
                            } else {
                                (jalpha[y][rd], ralpha[y][rd])
                            };
                            ra[idx] = flogsum(ra[idx], jc + tsc_v[yo]);
                            ra[idx] = flogsum(ra[idx], rc + tsc_v[yo]);
                        }
                        ra[idx] = ra[idx].max(IMPOSSIBLE);
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            if !state_is_detached(cm, v) {
                let lyoffset0 = if stt == IR_ST { 1 } else { 0 };
                for j in 1..=w {
                    let jm1 = (j - 1) * stride;
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let dm1 = jm1 + (d - 1);
                        for yo in 0..cnum {
                            let y = cfirst + yo;
                            let (jc, rc) = if y == v {
                                (ja[dm1], ra[dm1])
                            } else {
                                (jalpha[y][dm1], ralpha[y][dm1])
                            };
                            ja[idx] = flogsum(ja[idx], jc + tsc_v[yo]);
                            ra[idx] = flogsum(ra[idx], rc + tsc_v[yo]);
                        }
                        let e = sing_sc(esc_v, dsq[j]);
                        ja[idx] = (ja[idx] + e).max(IMPOSSIBLE);
                        ra[idx] = if d >= 2 { ra[idx] + e } else { e };
                        ra[idx] = ra[idx].max(IMPOSSIBLE);
                        let ld = j * stride + d;
                        for yo in lyoffset0..cnum {
                            let y = cfirst + yo;
                            let (jc, lc) = if y == v {
                                (ja[ld], la[ld])
                            } else {
                                (jalpha[y][ld], lalpha[y][ld])
                            };
                            la[idx] = flogsum(la[idx], jc + tsc_v[yo]);
                            la[idx] = flogsum(la[idx], lc + tsc_v[yo]);
                        }
                        la[idx] = la[idx].max(IMPOSSIBLE);
                    }
                }
            }
        } else if stt == MP_ST {
            for yo in 0..cnum {
                let y = cfirst + yo;
                let tsc = tsc_v[yo];
                for j in 1..=w {
                    let jm1 = (j - 1) * stride;
                    for d in 2..=j {
                        let idx = j * stride + d;
                        ja[idx] = flogsum(ja[idx], jalpha[y][jm1 + (d - 2)] + tsc);
                    }
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let src = j * stride + (d - 1);
                        la[idx] = flogsum(la[idx], jalpha[y][src] + tsc);
                        la[idx] = flogsum(la[idx], lalpha[y][src] + tsc);
                    }
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let src = jm1 + (d - 1);
                        ra[idx] = flogsum(ra[idx], jalpha[y][src] + tsc);
                        ra[idx] = flogsum(ra[idx], ralpha[y][src] + tsc);
                    }
                }
            }
            let lmv = &lm[v];
            let rmv = &rm[v];
            for j in 0..=w {
                let idx1 = j * stride + 1;
                ja[idx1] = IMPOSSIBLE;
                if j >= 1 {
                    la[idx1] = marg_sc(lmv, dsq[j]);
                    ra[idx1] = marg_sc(rmv, dsq[j]);
                }
                let mut i = if j >= 1 { j - 1 } else { 0 };
                for d in 2..=j {
                    let idx = j * stride + d;
                    ja[idx] += pair_sc(esc_v, dsq[i], dsq[j]);
                    la[idx] += marg_sc(lmv, dsq[i]);
                    ra[idx] += marg_sc(rmv, dsq[j]);
                    i = i.wrapping_sub(1);
                }
            }
            for j in 0..=w {
                for d in 1..=j {
                    let idx = j * stride + d;
                    ja[idx] = ja[idx].max(IMPOSSIBLE);
                    la[idx] = la[idx].max(IMPOSSIBLE);
                    ra[idx] = ra[idx].max(IMPOSSIBLE);
                }
            }
        } else if stt != B_ST {
            for yo in 0..cnum {
                let y = cfirst + yo;
                let tsc = tsc_v[yo];
                for j in 0..=w {
                    let jrow = j * stride;
                    for d in 0..=j {
                        let idx = jrow + d;
                        let src = jrow + d;
                        ja[idx] = flogsum(ja[idx], jalpha[y][src] + tsc);
                        la[idx] = flogsum(la[idx], lalpha[y][src] + tsc);
                        ra[idx] = flogsum(ra[idx], ralpha[y][src] + tsc);
                    }
                    la[jrow] = IMPOSSIBLE;
                    ra[jrow] = IMPOSSIBLE;
                }
            }
        } else {
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            for j in 0..=w {
                let jrow = j * stride;
                for d in 0..=j {
                    let idx = jrow + d;
                    for k in 0..=d {
                        let ly_src = (j - k) * stride + (d - k);
                        let z_src = jrow + k;
                        ja[idx] = flogsum(ja[idx], jalpha[y][ly_src] + jalpha[z][z_src]);
                        la[idx] = flogsum(la[idx], jalpha[y][ly_src] + lalpha[z][z_src]);
                        ra[idx] = flogsum(ra[idx], ralpha[y][ly_src] + jalpha[z][z_src]);
                    }
                    for k in 1..d {
                        let ly_src = (j - k) * stride + (d - k);
                        let z_src = jrow + k;
                        ta[idx] = flogsum(ta[idx], ralpha[y][ly_src] + lalpha[z][z_src]);
                    }
                    la[idx] = flogsum(la[idx], jalpha[y][idx]);
                    la[idx] = flogsum(la[idx], lalpha[y][idx]);
                    ra[idx] = flogsum(ra[idx], jalpha[z][idx]);
                    ra[idx] = flogsum(ra[idx], ralpha[z][idx]);
                }
            }
        }

        let trpenalty = pty[v];
        if not_impossible(trpenalty) {
            let root = w * stride + w;
            j0 = flogsum(j0, ja[root] + trpenalty);
            l0 = flogsum(l0, la[root] + trpenalty);
            r0 = flogsum(r0, ra[root] + trpenalty);
            if stt == B_ST {
                t0 = flogsum(t0, ta[root] + trpenalty);
            }
        }

        jalpha[v] = ja;
        lalpha[v] = la;
        ralpha[v] = ra;
        if stt == B_ST {
            talpha[v] = ta;
        }
    }

    j0.max(l0).max(r0).max(t0)
}

/// C `ParsetreeToCMBounds` (cm_parsetree.c:2611). Returns
/// (cfrom_span, cto_span, cfrom_emit, cto_emit). `have_i0`/`have_j0` = whether the
/// parse spans the first/last residue of the source sequence (TRUE for whole-seq
/// alignment). `pass_idx` selects the span-guess logic.
pub fn parsetree_to_cm_bounds(
    cm: &CM,
    emap: &EmitMap,
    tr: &Parsetree,
    pass_idx: i32,
    have_i0: bool,
    have_j0: bool,
) -> (i32, i32, i32, i32) {
    let clen = cm.clen;
    let mut cfrom_emit = clen + 1;
    let mut cto_emit = 0;

    let node_lpos = |nd: usize| -> i32 {
        let ndt = cm.ndtype[nd] as i32;
        if ndt == MATP_ND || ndt == MATL_ND {
            emap.lpos[nd]
        } else {
            emap.lpos[nd] + 1
        }
    };
    let node_rpos = |nd: usize| -> i32 {
        let ndt = cm.ndtype[nd] as i32;
        if ndt == MATP_ND || ndt == MATR_ND {
            emap.rpos[nd]
        } else {
            emap.rpos[nd] - 1
        }
    };

    for ti in 0..tr.n as usize {
        let v = tr.state[ti];
        if v == cm.m {
            continue; // EL: not produced on this global path
        }
        let vu = v as usize;
        let stt = cm.sttype[vu] as i32;
        let nd = cm.ndidx[vu] as usize;
        let ndt = cm.ndtype[nd] as i32;
        let mode = tr.mode[ti];
        let lpos = node_lpos(nd);
        let rpos = node_rpos(nd);

        let (is_left, is_right, insert_sd) = if stt == IL_ST {
            (true, false, 1)
        } else if stt == IR_ST {
            (false, true, 1)
        } else if ndt == MATP_ND {
            (true, true, 0)
        } else if ndt == MATL_ND {
            (true, false, 0)
        } else if ndt == MATR_ND {
            (false, true, 0)
        } else {
            (false, false, 0)
        };
        let emits_left = mode == TRMODE_J || mode == TRMODE_L;
        let emits_right = mode == TRMODE_J || mode == TRMODE_R;
        if is_left && emits_left {
            cfrom_emit = cfrom_emit.min(lpos + insert_sd);
            cto_emit = cto_emit.max(lpos);
        }
        if is_right && emits_right {
            cfrom_emit = cfrom_emit.min(rpos + insert_sd);
            cto_emit = cto_emit.max(rpos);
        }
    }

    // span (guess at full-parse boundaries), from the entry node (state[1]).
    let nd0 = cm.ndidx[tr.state[1] as usize] as usize;
    let mut cfrom_span = node_lpos(nd0);
    let mut cto_span = node_rpos(nd0);
    let mut nd = nd0 as i32;

    if pass_idx == PLI_PASS_5P_ONLY_FORCE && have_i0 {
        let target = cto_span;
        let mut rpos = target;
        while rpos == target && nd > 0 {
            nd -= 1;
            rpos = node_rpos(nd as usize);
        }
        cfrom_span = node_lpos(nd as usize);
    }
    if pass_idx == PLI_PASS_3P_ONLY_FORCE && have_j0 {
        let target = cfrom_span;
        let mut lpos = target;
        while lpos == target && nd > 0 {
            nd -= 1;
            lpos = node_lpos(nd as usize);
        }
        cto_span = node_rpos(nd as usize);
    }
    if pass_idx == PLI_PASS_5P_AND_3P_FORCE && have_i0 && have_j0 {
        cfrom_span = 1;
        cto_span = clen;
    }

    (cfrom_span, cto_span, cfrom_emit, cto_emit)
}

/// Truncation classification string from the span/emit bounds (C
/// `cm_alidisplay_TruncString`).
pub fn trunc_string(cfrom_span: i32, cto_span: i32, cfrom_emit: i32, cto_emit: i32) -> &'static str {
    let is5 = cfrom_emit != cfrom_span;
    let is3 = cto_emit != cto_span;
    if is5 && is3 {
        "5'&3'"
    } else if is5 {
        "5'"
    } else if is3 {
        "3'"
    } else {
        "no"
    }
}

/// Build the per-MP marginal emission tables (public wrapper).
pub fn marginal_emissions(cm: &CM) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    build_marginal_emissions(cm)
}
