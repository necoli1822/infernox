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
    B_ST, D_ST, E_ST, EL_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST,
    BEGL_S, BEGR_S,
    MATP_ND, MATL_ND, MATR_ND, BIF_ND, BEGL_ND, BEGR_ND, ROOT_ND, END_ND,
    MATP_MP, MATL_ML, MATR_MR, BIF_B, MATP_IL, END_E,
    MATL_D, MATL_IL, MATP_ML, MATP_MR, MATP_D, MATP_IR, MATR_D, MATR_IR,
    BEGR_IL, ROOT_S, ROOT_IL, ROOT_IR, EL,
};
use crate::cp9::EmitMap;
use crate::parsetree::Parsetree;

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
pub const PLI_PASS_5P_AND_3P_ANY: i32 = 5;
/// C `PLI_PASS_HMM_ONLY_ANY` (infernal.h:2103): the HMM-only pipeline pass.
pub const PLI_PASS_HMM_ONLY_ANY: i32 = 6;

// CM configuration flags (infernal.h:1937-1938).
const CMH_LOCAL_BEGIN: u32 = 1 << 10;
const CMH_LOCAL_END: u32 = 1 << 11;

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
        PLI_PASS_5P_AND_3P_ANY => Some(TRPENALTY_5P_AND_3P),
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
    // C StateIsDetached (cm.c:1777): cm->stid[v+1] == END_E. C allocates stid[M+1]
    // with the sentinel cm->stid[cm->M] = END_EL (cm.c:272), which is != END_E, so
    // StateIsDetached(M-1) is FALSE. infernox's stid has only M entries, so replicate
    // the sentinel: v+1 == M (the EL index) is never a detached insert.
    if v + 1 >= cm.m as usize {
        return false;
    }
    (cm.stid[v + 1] as i32) == END_E
}

/// GLOBAL truncated-begin penalty arrays (C `trp->g_ptyAA`), log2, per state.
/// `g[idx][v]` is the truncated-begin penalty for entering state `v` under pass
/// index `idx` (5P_AND_3P / 5P_ONLY / 3P_ONLY). IMPOSSIBLE = disallowed.
pub struct TrPenalties {
    pub g: [Vec<f32>; 3],
    /// Integer (scaled) global truncated-begin penalties (C `trp->ig_ptyAA`).
    /// `ig[idx][v] = Prob2Score(prob, 1.0)`; -INFTY where disallowed.
    pub ig: [Vec<i32>; 3],
    /// LOCAL-mode truncated-begin penalty arrays (C `trp->l_ptyAA`), log2, per state.
    /// Used by the DP when CMH_LOCAL_BEGIN is set; global mode uses `g`.
    pub l: [Vec<f32>; 3],
    /// Integer (scaled) local penalties (C `trp->il_ptyAA`).
    pub il: [Vec<i32>; 3],
}

impl TrPenalties {
    /// C `cm_TrCYKInsideAlign` / etc. penalty selection: `(cm->flags &
    /// CMH_LOCAL_BEGIN) ? l_ptyAA[pty] : g_ptyAA[pty]`. Returns the log2 penalty
    /// slice for a pass index in the requested (local vs global) configuration.
    pub fn pty_slice(&self, pass_idx: i32, local: bool) -> &[f32] {
        let idx = pty_idx_for_pass(pass_idx).expect("truncated pass");
        if local { self.l[idx].as_slice() } else { self.g[idx].as_slice() }
    }
}

const CP9_INFTY: i32 = 987654321;

/// C `sreLOG2(x)` (infernal.h:153): log(x)*1.44269504 for x>0, else IMPOSSIBLE.
#[inline]
fn sre_log2(x: f64) -> f64 {
    if x > 0.0 {
        x.ln() * 1.44269504
    } else {
        -1e36
    }
}

/// C `Prob2Score(p, null)` (cm.c): floor(0.5 + 1000*sreLOG2(p/null)); -INFTY if p==0.
#[inline]
fn prob2score(p: f32, null: f32) -> i32 {
    if p == 0.0 {
        -CP9_INFTY
    } else {
        (0.5_f64 + 1000.0_f64 * sre_log2((p / null) as f64)).floor() as i32
    }
}

impl TrPenalties {
    /// Port of `cm_tr_penalties_Create` (cm_trunc.c:51) with `ignore_inserts=FALSE`
    /// (the value used by cm_Configure). Fills the GLOBAL (`g`/`ig`) and LOCAL
    /// (`l`/`il`) truncated-begin penalty arrays.
    /// `psi` = expected state occupancy (cm_expected_state_occupancy of the same
    /// global-config CM; matches C's cm_ExpectedPositionOccupancy psi). `emap` = its
    /// emit map.
    pub fn new(cm: &CM, emap: &EmitMap, psi: &[f64]) -> Self {
        let m = cm.m as usize;
        let clen = cm.clen;
        let mut g: [Vec<f32>; 3] = [
            vec![IMPOSSIBLE; m],
            vec![IMPOSSIBLE; m],
            vec![IMPOSSIBLE; m],
        ];
        let mut ig: [Vec<i32>; 3] = [
            vec![-CP9_INFTY; m],
            vec![-CP9_INFTY; m],
            vec![-CP9_INFTY; m],
        ];
        let mut l: [Vec<f32>; 3] = [
            vec![IMPOSSIBLE; m],
            vec![IMPOSSIBLE; m],
            vec![IMPOSSIBLE; m],
        ];
        let mut il: [Vec<i32>; 3] = [
            vec![-CP9_INFTY; m],
            vec![-CP9_INFTY; m],
            vec![-CP9_INFTY; m],
        ];
        // C cm_trunc.c:289-290: `g_5and3 = 2. / (cm->clen * (cm->clen+1))` and
        // `g_5or3 = 1. / cm->clen`. cm->clen is int; the denominators are int, but
        // the `2.`/`1.` literals are doubles, so both divisions are done in DOUBLE
        // and the (double) result is stored into a `float`. Reproduce the exact
        // double-then-truncate (not an f32 division).
        let g_5and3 = (2.0f64 / ((clen * (clen + 1)) as f64)) as f32;
        let g_5or3 = (1.0f64 / clen as f64) as f32;

        // C cm_trunc.c:280-281: begin[] = cm_CalculateLocalBeginProbs(cm, cm->pbegin,
        // cm->t, begin). Compute inline with C-exact float ops (cm_modelconfig.c:397):
        // node 1 gets 1-pbegin (double subtract -> float), internal MAT*/BIF nodes
        // share pbegin/(float)nstarts computed in FLOAT (not f64).
        let mut begin = vec![0.0f32; m];
        let mut nstarts = 0i32;
        for nd in 2..cm.nodes as usize {
            let t = cm.ndtype[nd] as i32;
            if t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BIF_ND {
                nstarts += 1;
            }
        }
        let pbegin = cm.pbegin; // f32, = cm->pbegin
        if cm.nodes > 1 {
            // C: begin[nodemap[1]] = 1. - p_internal_start;  (1. is double)
            begin[cm.nodemap[1] as usize] = (1.0f64 - pbegin as f64) as f32;
        }
        // C: p = p_internal_start / (float) nstarts;  (FLOAT division)
        let p_begin = pbegin / nstarts as f32;
        for nd in 2..cm.nodes as usize {
            let t = cm.ndtype[nd] as i32;
            if t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BIF_ND {
                begin[cm.nodemap[nd] as usize] = p_begin;
            }
        }

        // C cm_trunc.c:298: prv5 = prv3 = prv53 = 0. (running local-penalty state).
        let mut prv5: f32 = 0.0;
        let mut prv3: f32 = 0.0;
        let mut prv53: f32 = 0.0;

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
            // C cm_trunc.c:304-311: reset/seed the running prv* values at END and
            // BEGL/BEGR nodes (BEG nodes seed from their parent BIF_B's local probs).
            if ndt == END_ND {
                prv5 = 0.0;
                prv3 = 0.0;
                prv53 = 0.0;
            } else if ndt == BEGL_ND || ndt == BEGR_ND {
                let bif_b = cm.plast[cm.nodemap[nd] as usize] as usize;
                prv5 = if ndt == BEGL_ND { 0.0 } else { l[TRPENALTY_5P_ONLY][bif_b] };
                prv3 = if ndt == BEGR_ND { 0.0 } else { l[TRPENALTY_3P_ONLY][bif_b] };
                prv53 = l[TRPENALTY_5P_AND_3P][bif_b];
            } else if ndt == MATP_ND || ndt == MATL_ND || ndt == MATR_ND || ndt == BIF_ND {
                let mst = cm.nodemap[nd] as usize;
                let (i1, i2) = inserts_given_node(cm, nd as i32 - 1);
                // C cm_trunc.c:311-315: m_psi, i1_psi, i2_psi, summed_psi are all
                // `float` (psi is `double`, truncated to float on assignment), and
                // the `(m_psi / summed_psi) * prob` ratio is computed entirely in
                // float — reproduce those float ops exactly (not an f64 divide).
                //   float m_psi = psi[m];
                //   if(cm->ndtype[nd] == MATP_MP) { m_psi += (psi[m+1] + psi[m+2]); }
                //
                // NOTE: the C condition is `cm->ndtype[nd] == MATP_MP`. `ndtype[nd]`
                // is a NODE type (MATP_nd=1, MATL_nd=2, ...) but `MATP_MP`=6 is a
                // STATE-id (unique statetype), not a node type. No node reaching this
                // branch has ndtype==6 (only ROOT_nd is 6, and ROOT is not a
                // MAT*/BIF node), so this test is ALWAYS false in C: the MATP ML/MR
                // occupancies are never added and m_psi stays psi[m] for every state.
                // This is evidently a C typo (intent was MATP_nd), but we transcribe
                // the exact behavior — reproducing intent here diverges by ~1e-4.
                let mut m_psi: f32 = psi[mst] as f32;
                if cm.ndtype[nd] as i32 == MATP_MP {
                    m_psi = (m_psi as f64 + (psi[mst + 1] + psi[mst + 2])) as f32;
                }
                //   float i1_psi = (i1==-1) ? 0. : psi[i1];  (double -> float)
                let i1_psi: f32 = if i1 >= 0 { psi[i1 as usize] as f32 } else { 0.0 };
                let i2_psi: f32 = if i2 >= 0 { psi[i2 as usize] as f32 } else { 0.0 };
                //   float summed_psi = m_psi + i1_psi + i2_psi;  (all float)
                let summed: f32 = m_psi + i1_psi + i2_psi;

                // C: trp->{g,l}_ptyAA[idx][state] = (state_psi / summed_psi) * prob; (float)
                let set = |g: &mut Vec<f32>, prob: f32| {
                    g[mst] = (m_psi / summed) * prob;
                    if i1 >= 0 {
                        g[i1 as usize] = (i1_psi / summed) * prob;
                    }
                    if i2 >= 0 {
                        g[i2 as usize] = (i2_psi / summed) * prob;
                    }
                };
                // Global penalties (C cm_trunc.c:338-352).
                set(&mut g[TRPENALTY_5P_AND_3P], g_5and3);
                if rpos == clen {
                    set(&mut g[TRPENALTY_5P_ONLY], g_5or3);
                }
                if lpos == 1 {
                    set(&mut g[TRPENALTY_3P_ONLY], g_5or3);
                }

                // Local penalties (C cm_trunc.c:355-388). All three set unconditionally.
                // C: subtree_clen = rpos - lpos + 1; nfrag5=nfrag3=subtree_clen;
                //    nfrag53 = (subtree_clen * (subtree_clen+1)) / 2;  (integer division)
                let subtree_clen = rpos - lpos + 1;
                let nfrag5 = subtree_clen;
                let nfrag3 = subtree_clen;
                let nfrag53 = (subtree_clen * (subtree_clen + 1)) / 2;
                // C: cur5 = begin[m] / (float) nfrag5 + prv5; (all float, div before add)
                let cur5 = begin[mst] / nfrag5 as f32 + prv5;
                let cur3 = begin[mst] / nfrag3 as f32 + prv3;
                let cur53 = begin[mst] / nfrag53 as f32 + prv53;
                set(&mut l[TRPENALTY_5P_AND_3P], cur53);
                set(&mut l[TRPENALTY_5P_ONLY], cur5);
                set(&mut l[TRPENALTY_3P_ONLY], cur3);

                // C cm_trunc.c:386-388: update running prv* for the next node.
                prv5 = if ndt == MATL_ND { cur5 } else { 0.0 };
                prv3 = if ndt == MATR_ND { cur3 } else { 0.0 };
                prv53 = cur53;
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
            // C cm_trunc.c:349-361: set integer penalty (Prob2Score of the prob)
            // then convert the float prob to a log2 penalty (sreLOG2), for each idx
            // whose prob is not IMPOSSIBLE.
            for idx in 0..3 {
                if not_impossible(g[idx][v]) {
                    ig[idx][v] = prob2score(g[idx][v], 1.0);
                    g[idx][v] = sre_log2(g[idx][v] as f64) as f32;
                }
            }
            // C cm_trunc.c:440-445: local penalties, all three converted unconditionally
            // (they are always set to a non-IMPOSSIBLE value for qualifying states).
            for idx in 0..3 {
                il[idx][v] = prob2score(l[idx][v], 1.0);
                l[idx][v] = sre_log2(l[idx][v] as f64) as f32;
            }
        }

        TrPenalties { g, ig, l, il }
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
/// Marginal emission accessor: C reads `lmesc_v[dsq[i]]` directly over the full
/// augmented alphabet (0..Kp). `m` is `cm.lmesc[v]` / `cm.rmesc[v]`.
#[inline]
fn marg_sc(m: &[f32], di: u8) -> f32 {
    m[di as usize]
}

/// Non-banded truncated CYK alignment of `dsq[1..=lp]` to the GLOBAL-config CM,
/// for pipeline pass `pass_idx` (one of the FORCE truncated passes). Fills the
/// J/L/R/T marginal matrices (preset_mode = UNKNOWN, best mode chosen by score),
/// traces back and returns (parsetree-with-modes, score, mode).
///
/// Port of `cm_TrCYKInsideAlign` + `cm_tr_alignT` (CYK path). `dsq[0]` is unused.
///
/// `use_local` selects the truncation-penalty arrays: C `cm_TrCYKInsideAlign` uses
/// `(cm->flags & CMH_LOCAL_BEGIN) ? l_ptyAA : g_ptyAA`. Callers that must preserve a
/// byte-verified global-penalty path (cmsearch) pass `false`.
pub fn tr_cyk_align(
    cm: &CM,
    trp: &TrPenalties,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    lp: i32,
    pass_idx: i32,
    use_local: bool,
) -> (Parsetree, f32, i8) {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let ncell = stride * stride;
    let pty = trp.pty_slice(pass_idx, use_local);

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

    // C cm_dpalign_trunc.c:1331-1333: precompute local-end scores el_scA[d] = el_selfsc*d
    // (score of an EL emit of d residues). Used to re-init decks for local ends below.
    let mut el_sca = vec![0.0f32; w + 1];
    for d in 0..=w {
        el_sca[d] = cm.el_selfsc * d as f32;
    }

    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];

        // local decks: init all IMPOSSIBLE, then (C cm_dpalign_trunc.c:1384-1405)
        // re-initialize the J/L/R decks if a local end (EL) is possible from v, i.e.
        // NOT_IMPOSSIBLE(cm->endsc[v]). Base score = el_scA[d-sd] + endsc[v]; the shadow
        // stays USED_EL (set below). fill_L = fill_R = TRUE here (preset_mode UNKNOWN).
        let mut ja = vec![IMPOSSIBLE; ncell];
        let mut la = vec![IMPOSSIBLE; ncell];
        let mut ra = vec![IMPOSSIBLE; ncell];
        let mut ta = vec![IMPOSSIBLE; ncell];
        let endsc_v = cm.endsc[v];
        if not_impossible(endsc_v) {
            let sd = tr_state_delta(stt) as usize;
            let sdl = tr_state_left_delta(stt) as usize;
            let sdr = tr_state_right_delta(stt) as usize;
            for j in 0..=w {
                for d in sd..=j {
                    ja[j * stride + d] = el_sca[d - sd] + endsc_v;
                }
                for d in sdl..=j {
                    la[j * stride + d] = el_sca[d - sdl] + endsc_v;
                }
                for d in sdr..=j {
                    ra[j * stride + d] = el_sca[d - sdr] + endsc_v;
                }
            }
        }
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
    tr.is_std = false; // C cm_tr_alignT: truncated parse
    tr.pass_idx = pass_idx;
    // C cm_dpalign_trunc.c:233: tr->trpenalty = (local?l:g)_ptyAA[pty_idx][b].
    tr.trpenalty = pty[b as usize];
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
            // C cm_dpalign_trunc.c:380-390: for a local end (USED_EL), insert the EL
            // node (cm->M) spanning [i,j] as the left child before swinging to EL. For
            // USED_TRUNC_END no node is added; both then set v = cm->M.
            if yoffset == USED_EL {
                let idx = tr.add_node_mode(i, j, cm.m, -1, -1, tr.n - 1, mode);
                tr.nxtl[(tr.n - 2) as usize] = idx;
            }
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

#[inline]
fn tr_state_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST => 2,
        x if x == ML_ST || x == MR_ST || x == IL_ST || x == IR_ST => 1,
        _ => 0,
    }
}
#[inline]
fn tr_state_left_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST || x == ML_ST || x == IL_ST => 1,
        _ => 0,
    }
}
#[inline]
fn tr_state_right_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST || x == MR_ST || x == IR_ST => 1,
        _ => 0,
    }
}

/// HMM-banded truncated CYK alignment + traceback. Faithful port of C
/// `cm_TrCYKInsideAlignHB` (cm_dpalign_trunc.c:1889, the DP fill over the marginal
/// J/L/R/T planes) followed by the CYK branch of `cm_tr_alignT_hb`
/// (cm_dpalign_trunc.c:471, `do_optacc == FALSE`). This is the aligner
/// `pli_align_hit` (cm_pipeline.c:4295) invokes via `DispatchSqAlignment` with flags
/// `CM_ALIGN_TRUNC | CYK | HBANDED` to fix a truncated hit's model bounds; the
/// pipeline default uses OPTACC but C's HB-CYK produces byte-identical mdl-from/mdl-to
/// (verified by `cmsearch -g --acyk`), and the non-truncated infernox path likewise
/// uses HB-CYK (`cyk_align_hb_cmbounds`) as the faithful bounds source.
///
/// `cp9b` must already be the search's F7 bands shifted to the `[1..=lp]` hit frame
/// via [`crate::cp9::shift_cm_bands_trunc`]. `dsq` is indexed so `dsq[1..=lp]` are the
/// hit residues (pass `&wdsq[(hi - 1)..]`). `preset_mode` is the hit's marginal mode
/// (C `hit->mode`); pass [`TRMODE_UNKNOWN`] to let the DP pick the max-scoring mode.
///
/// Returns `(parsetree, score, mode)`, drop-in compatible with [`tr_cyk_align`].
/// GLOBAL config only (matching the `-g` truncated pipeline): no local begins/ends/EL.
/// Falls back to the non-banded [`tr_cyk_align`] if the shifted bands cannot admit a
/// full alignment to ROOT_S (C errors + `pli_align_hit` fails over, but the mdl bounds
/// then still come from a truncated CYK parse).
#[allow(clippy::too_many_arguments)]
pub fn tr_cyk_align_hb(
    cm: &CM,
    trp: &TrPenalties,
    cp9b: &crate::cp9::CP9Bands,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    lp: i32,
    pass_idx: i32,
    preset_mode: i8,
    use_local: bool,
) -> (Parsetree, f32, i8) {
    let m = cm.m as usize;
    let l = lp;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let kp = crate::cm::ALPHABET_SIZE_P;

    // C cm_TrFillFromMode (cm_dpalign_trunc.c:9858): which of L/R/T decks to fill.
    let (fill_l, fill_r, fill_t) = match preset_mode {
        TRMODE_J => (false, false, false),
        TRMODE_L => (true, false, false),
        TRMODE_R => (false, true, false),
        _ => (true, true, true), // TRMODE_T or TRMODE_UNKNOWN
    };

    let pty = trp.pty_slice(pass_idx, use_local);

    // Validate a full alignment to ROOT_S (v==0) is admitted by the shifted bands
    // (C cm_TrCYKInsideAlignHB entry checks). Fall back to non-banded TrCYK otherwise.
    if jmin[0] > l || jmax[0] < l {
        return tr_cyk_align(cm, trp, lm, rm, dsq, lp, pass_idx, use_local);
    }
    let jp_0 = (l - jmin[0]) as usize;
    if hdmin[0][jp_0] > l || hdmax[0][jp_0] < l {
        return tr_cyk_align(cm, trp, lm, rm, dsq, lp, pass_idx, use_local);
    }
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    // ---- allocate banded {J,L,R,T} DP decks + shadow decks (C cm_tr_hb_mx_GrowTo) ----
    let njv = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut jalpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
    let mut lalpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
    let mut ralpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
    let mut talpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
    let mut jysh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut lysh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut rysh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut jksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut lksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut rksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut tksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut lkmode: Vec<Vec<Vec<i8>>> = Vec::with_capacity(m);
    let mut rkmode: Vec<Vec<Vec<i8>>> = Vec::with_capacity(m);
    for v in 0..m {
        let nj = njv(v);
        let is_b = cm.sttype[v] as i32 == B_ST;
        let mut jd = Vec::with_capacity(nj);
        let mut ld = Vec::with_capacity(nj);
        let mut rd = Vec::with_capacity(nj);
        let mut td = Vec::with_capacity(nj);
        let mut jys = Vec::with_capacity(nj);
        let mut lys = Vec::with_capacity(nj);
        let mut rys = Vec::with_capacity(nj);
        let mut jks = Vec::with_capacity(nj);
        let mut lks = Vec::with_capacity(nj);
        let mut rks = Vec::with_capacity(nj);
        let mut tks = Vec::with_capacity(nj);
        let mut lkm = Vec::with_capacity(nj);
        let mut rkm = Vec::with_capacity(nj);
        for jp in 0..nj {
            let w = (hdmax[v][jp] - hdmin[v][jp] + 1).max(0) as usize;
            jd.push(vec![IMPOSSIBLE; w]);
            ld.push(vec![IMPOSSIBLE; w]);
            rd.push(vec![IMPOSSIBLE; w]);
            jys.push(vec![USED_EL; w]);
            lys.push(vec![USED_EL; w]);
            rys.push(vec![USED_EL; w]);
            // C cm_mx.c:600,646 — the T (Terminal) deck is allocated for B states
            // AND the ROOT_S (v==0), whose [L][L] cell records the optimal T-mode
            // truncated-begin score. (Tkshadow[0] is NOT allocated — cm_dpalign_trunc.c
            // :2853 "Tyshadow[0] doesn't exist, caller must know how to deal".)
            if is_b || v == 0 {
                td.push(vec![IMPOSSIBLE; w]);
            }
            if is_b {
                jks.push(vec![0i32; w]);
                lks.push(vec![0i32; w]);
                rks.push(vec![0i32; w]);
                tks.push(vec![0i32; w]);
                lkm.push(vec![TRMODE_J; w]);
                rkm.push(vec![TRMODE_J; w]);
            }
        }
        jalpha.push(jd);
        lalpha.push(ld);
        ralpha.push(rd);
        talpha.push(td);
        jysh.push(jys);
        lysh.push(lys);
        rysh.push(rys);
        jksh.push(jks);
        lksh.push(lks);
        rksh.push(rks);
        tksh.push(tks);
        lkmode.push(lkm);
        rkmode.push(rkm);
    }

    // C cm_dpalign_trunc.c:1988: el_scA[d] = el_selfsc*d, for local-end deck re-init.
    let mut el_sca = vec![0.0f32; (l + 1) as usize];
    for d in 0..=l as usize {
        el_sca[d] = cm.el_selfsc * d as f32;
    }

    let mut jb = 0i32;
    let mut lb = 0i32;
    let mut rb = 0i32;
    let mut tb = 0i32;

    // ---- Main recursion: for (v = M-1; v > 0; v--) (cm_dpalign_trunc.c:2027) ----
    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = tr_state_delta(stt);
        let sdl = tr_state_left_delta(stt);
        let sdr = tr_state_right_delta(stt);
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let lmesc_v = &lm[v];
        let rmesc_v = &rm[v];
        let do_j_v = cp9b.jvalid[v];
        let do_l_v = cp9b.lvalid[v] && fill_l;
        let do_r_v = cp9b.rvalid[v] && fill_r;
        let do_t_v = cp9b.tvalid[v] && fill_t;

        // C cm_dpalign_trunc.c:2050-2104: re-initialize the J/L/R decks if a local end
        // (EL) is possible from v (NOT_IMPOSSIBLE(cm->endsc[v])). Base score is
        // Jalpha[cm->M][j][d-sd] + endsc[v]; since we don't materialize the EL deck we
        // use the equivalent el_scA[d-sd] + endsc[v] (see C comment at :2065-2068).
        // Gated per-plane by do_{J,L,R}_v && cp9b->{J,L,R}valid[cm->M]. (global -g:
        // endsc[v] is IMPOSSIBLE everywhere, so this is a no-op.)
        let endsc_v = cm.endsc[v];
        if not_impossible(endsc_v) {
            if do_j_v && cp9b.jvalid[m] {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sd {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sd, (sd - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        jalpha[v][jp_v][dp_v] = el_sca[(d - sd) as usize] + endsc_v;
                        dp_v += 1;
                        d += 1;
                    }
                }
            }
            if do_l_v && cp9b.lvalid[m] {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdl {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sdl, (sdl - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        lalpha[v][jp_v][dp_v] = el_sca[(d - sdl) as usize] + endsc_v;
                        dp_v += 1;
                        d += 1;
                    }
                }
            }
            if do_r_v && cp9b.rvalid[m] {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdr {
                        (hdmin[v][jp_v], 0usize)
                    } else {
                        (sdr, (sdr - hdmin[v][jp_v]) as usize)
                    };
                    while d <= hdmax[v][jp_v] {
                        ralpha[v][jp_v][dp_v] = el_sca[(d - sdr) as usize] + endsc_v;
                        dp_v += 1;
                        d += 1;
                    }
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                if do_j_v { jalpha[v][jp_v][0] = 0.0; }
                if do_l_v { lalpha[v][jp_v][0] = 0.0; }
                if do_r_v { ralpha[v][jp_v][0] = 0.0; }
            }
        } else if stt == IL_ST || stt == ML_ST {
            if !state_is_detached(cm, v) {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let j_sdr = j - sdr;
                    // valid children (j-sdr within y's j band)
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        if j_sdr >= jmin[y] && j_sdr <= jmax[y] { yvalid.push(yoffset); }
                    }
                    for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                        let i = j - d + 1;
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        // Handle J and L first (cm_dpalign_trunc.c:2144)
                        if do_j_v || do_l_v {
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_j_y = cp9b.jvalid[y];
                                let do_l_y = cp9b.lvalid[y] && fill_l;
                                if do_j_y || do_l_y {
                                    let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                    if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                        let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                        if do_j_v && do_j_y {
                                            let sc = jalpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset];
                                            if sc > jalpha[v][jp_v][dp_v] {
                                                jalpha[v][jp_v][dp_v] = sc;
                                                jysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET;
                                            }
                                        }
                                        if do_l_v && do_l_y {
                                            let sc = lalpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset];
                                            if sc > lalpha[v][jp_v][dp_v] {
                                                lalpha[v][jp_v][dp_v] = sc;
                                                lysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_L_OFFSET;
                                            }
                                        }
                                    }
                                }
                            }
                            if do_j_v {
                                jalpha[v][jp_v][dp_v] += esc_v[dsq[i as usize] as usize];
                                jalpha[v][jp_v][dp_v] = jalpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                            if do_l_v {
                                if d >= 2 {
                                    lalpha[v][jp_v][dp_v] += esc_v[dsq[i as usize] as usize];
                                } else {
                                    lalpha[v][jp_v][dp_v] = esc_v[dsq[i as usize] as usize];
                                    lysh[v][jp_v][dp_v] = USED_TRUNC_END;
                                }
                                lalpha[v][jp_v][dp_v] = lalpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                        }
                        if do_r_v {
                            // Handle R separately (cm_dpalign_trunc.c:2186); uses d (not d-sd)
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_r_y = cp9b.rvalid[y] && fill_r;
                                let do_j_y = cp9b.jvalid[y];
                                if (do_j_y || do_r_y) && y != v {
                                    let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                    if d >= hdmin[y][jp_y_sdr] && d <= hdmax[y][jp_y_sdr] {
                                        let dp_y = (d - hdmin[y][jp_y_sdr]) as usize;
                                        if do_j_y {
                                            let sc = jalpha[y][jp_y_sdr][dp_y] + tsc_v[yoffset];
                                            if sc > ralpha[v][jp_v][dp_v] {
                                                ralpha[v][jp_v][dp_v] = sc;
                                                rysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET;
                                            }
                                        }
                                        if do_r_y {
                                            let sc = ralpha[y][jp_y_sdr][dp_y] + tsc_v[yoffset];
                                            if sc > ralpha[v][jp_v][dp_v] {
                                                ralpha[v][jp_v][dp_v] = sc;
                                                rysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_R_OFFSET;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            if !state_is_detached(cm, v) {
                // First loop: J and R matrices (same j set) (cm_dpalign_trunc.c:2246)
                if do_j_v || do_r_v {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let j_sdr = j - sdr;
                        let mut yvalid: Vec<usize> = Vec::new();
                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            if j_sdr >= jmin[y] && j_sdr <= jmax[y] { yvalid.push(yoffset); }
                        }
                        for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                            let dp_v = (d - hdmin[v][jp_v]) as usize;
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_j_y = cp9b.jvalid[y];
                                let do_r_y = cp9b.rvalid[y] && fill_r;
                                if do_j_y || do_r_y {
                                    let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                    if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                        let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                        if do_j_v && do_j_y {
                                            let sc = jalpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset];
                                            if sc > jalpha[v][jp_v][dp_v] {
                                                jalpha[v][jp_v][dp_v] = sc;
                                                jysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET;
                                            }
                                        }
                                        if do_r_v && do_r_y {
                                            let sc = ralpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset];
                                            if sc > ralpha[v][jp_v][dp_v] {
                                                ralpha[v][jp_v][dp_v] = sc;
                                                rysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_R_OFFSET;
                                            }
                                        }
                                    }
                                }
                            }
                            if do_j_v {
                                jalpha[v][jp_v][dp_v] += esc_v[dsq[j as usize] as usize];
                                jalpha[v][jp_v][dp_v] = jalpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                            if do_r_v {
                                if d >= 2 {
                                    ralpha[v][jp_v][dp_v] += esc_v[dsq[j as usize] as usize];
                                } else {
                                    ralpha[v][jp_v][dp_v] = esc_v[dsq[j as usize] as usize];
                                    rysh[v][jp_v][dp_v] = USED_TRUNC_END;
                                }
                                ralpha[v][jp_v][dp_v] = ralpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                        }
                    }
                }
                // Second loop: L matrix (different j set: uses j, not j-sdr) (:2298)
                if do_l_v {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let mut yvalid: Vec<usize> = Vec::new();
                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            if j >= jmin[y] && j <= jmax[y] { yvalid.push(yoffset); }
                        }
                        for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                            let dp_v = (d - hdmin[v][jp_v]) as usize;
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_l_y = cp9b.lvalid[y] && fill_l;
                                let do_j_y = cp9b.jvalid[y];
                                if (do_j_y || do_l_y) && y != v {
                                    let jp_y = (j - jmin[y]) as usize;
                                    if d >= hdmin[y][jp_y] && d <= hdmax[y][jp_y] {
                                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                                        if do_j_y {
                                            let sc = jalpha[y][jp_y][dp_y] + tsc_v[yoffset];
                                            if sc > lalpha[v][jp_v][dp_v] {
                                                lalpha[v][jp_v][dp_v] = sc;
                                                lysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET;
                                            }
                                        }
                                        if do_l_y {
                                            let sc = lalpha[y][jp_y][dp_y] + tsc_v[yoffset];
                                            if sc > lalpha[v][jp_v][dp_v] {
                                                lalpha[v][jp_v][dp_v] = sc;
                                                lysh[v][jp_v][dp_v] = yoffset as i32 + TRMODE_L_OFFSET;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if stt == MP_ST {
            // MP cannot self-transit: for y { for j { for d } } (cm_dpalign_trunc.c:2366)
            for y in cfirst..(cfirst + cnum) {
                let do_j_y = cp9b.jvalid[y];
                let do_l_y = cp9b.lvalid[y] && fill_l;
                let do_r_y = cp9b.rvalid[y] && fill_r;
                let yoffset = (y - cfirst) as i32;
                let tsc = tsc_v[yoffset as usize];

                // J and R matrices (j set: j-sdr in y's band)
                if (do_j_v && do_j_y) || (do_r_v && (do_j_y || do_r_y)) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = (jn - jmin[v]) as i32;
                    let jpx = (jx - jmin[v]) as i32;
                    let mut jp_y_sdr = (jn - jmin[y] - sdr) as i32;
                    while jp_v <= jpx {
                        let jp_v_u = jp_v as usize;
                        let jp_y_sdr_u = jp_y_sdr as usize;
                        if do_j_v && do_j_y {
                            let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_sdr_u] + sd);
                            let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_sdr_u] + sd);
                            let mut dp_v = (dn - hdmin[v][jp_v_u]) as i32;
                            let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                            let mut dp_y_sd = (dn - hdmin[y][jp_y_sdr_u] - sd) as i32;
                            while dp_v <= dpx {
                                let sc = jalpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc;
                                if sc > jalpha[v][jp_v_u][dp_v as usize] {
                                    jalpha[v][jp_v_u][dp_v as usize] = sc;
                                    jysh[v][jp_v_u][dp_v as usize] = yoffset + TRMODE_J_OFFSET;
                                }
                                dp_v += 1;
                                dp_y_sd += 1;
                            }
                        }
                        if do_r_v && (do_r_y || do_j_y) {
                            let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_sdr_u] + sdr);
                            let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_sdr_u] + sdr);
                            let mut dp_v = (dn - hdmin[v][jp_v_u]) as i32;
                            let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                            let mut dp_y_sdr = (dn - hdmin[y][jp_y_sdr_u] - sdr) as i32;
                            while dp_v <= dpx {
                                if do_j_y {
                                    let sc = jalpha[y][jp_y_sdr_u][dp_y_sdr as usize] + tsc;
                                    if sc > ralpha[v][jp_v_u][dp_v as usize] {
                                        ralpha[v][jp_v_u][dp_v as usize] = sc;
                                        rysh[v][jp_v_u][dp_v as usize] = yoffset + TRMODE_J_OFFSET;
                                    }
                                }
                                if do_r_y {
                                    let sc = ralpha[y][jp_y_sdr_u][dp_y_sdr as usize] + tsc;
                                    if sc > ralpha[v][jp_v_u][dp_v as usize] {
                                        ralpha[v][jp_v_u][dp_v as usize] = sc;
                                        rysh[v][jp_v_u][dp_v as usize] = yoffset + TRMODE_R_OFFSET;
                                    }
                                }
                                dp_v += 1;
                                dp_y_sdr += 1;
                            }
                        }
                        jp_v += 1;
                        jp_y_sdr += 1;
                    }
                }
                // L matrix (j set: j in y's band) (cm_dpalign_trunc.c:2436)
                if do_l_v && (do_l_y || do_j_y) {
                    let jn = jmin[v].max(jmin[y]);
                    let jx = jmax[v].min(jmax[y]);
                    let mut jp_v = (jn - jmin[v]) as i32;
                    let jpx = (jx - jmin[v]) as i32;
                    let mut jp_y = (jn - jmin[y]) as i32;
                    while jp_v <= jpx {
                        let jp_v_u = jp_v as usize;
                        let jp_y_u = jp_y as usize;
                        let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_u] + sdl);
                        let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_u] + sdl);
                        let mut dp_v = (dn - hdmin[v][jp_v_u]) as i32;
                        let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                        let mut dp_y_sdl = (dn - hdmin[y][jp_y_u] - sdl) as i32;
                        while dp_v <= dpx {
                            if do_j_y {
                                let sc = jalpha[y][jp_y_u][dp_y_sdl as usize] + tsc;
                                if sc > lalpha[v][jp_v_u][dp_v as usize] {
                                    lalpha[v][jp_v_u][dp_v as usize] = sc;
                                    lysh[v][jp_v_u][dp_v as usize] = yoffset + TRMODE_J_OFFSET;
                                }
                            }
                            if do_l_y {
                                let sc = lalpha[y][jp_y_u][dp_y_sdl as usize] + tsc;
                                if sc > lalpha[v][jp_v_u][dp_v as usize] {
                                    lalpha[v][jp_v_u][dp_v as usize] = sc;
                                    lysh[v][jp_v_u][dp_v as usize] = yoffset + TRMODE_L_OFFSET;
                                }
                            }
                            dp_v += 1;
                            dp_y_sdl += 1;
                        }
                        jp_v += 1;
                        jp_y += 1;
                    }
                }
            }
            // add MP emission scores (cm_dpalign_trunc.c:2490)
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut i = j - hdmin[v][jp_v] + 1;
                let mut dp_v = 0usize;
                for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                    if d >= 2 {
                        if do_j_v {
                            let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                            jalpha[v][jp_v][dp_v] += esc_v[idx];
                        }
                        if do_l_v { lalpha[v][jp_v][dp_v] += lmesc_v[dsq[i as usize] as usize]; }
                        if do_r_v { ralpha[v][jp_v][dp_v] += rmesc_v[dsq[j as usize] as usize]; }
                    } else {
                        if do_j_v { jalpha[v][jp_v][dp_v] = IMPOSSIBLE; }
                        if do_l_v {
                            lalpha[v][jp_v][dp_v] = lmesc_v[dsq[i as usize] as usize];
                            lysh[v][jp_v][dp_v] = USED_TRUNC_END;
                        }
                        if do_r_v {
                            ralpha[v][jp_v][dp_v] = rmesc_v[dsq[j as usize] as usize];
                            rysh[v][jp_v][dp_v] = USED_TRUNC_END;
                        }
                    }
                    i -= 1;
                    dp_v += 1;
                }
            }
            // ensure all cells >= IMPOSSIBLE
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let ww = (hdmax[v][jp_v] - hdmin[v][jp_v]).max(-1);
                for dp_v in 0..=ww {
                    if ww < 0 { break; }
                    let dp = dp_v as usize;
                    if do_j_v { jalpha[v][jp_v][dp] = jalpha[v][jp_v][dp].max(IMPOSSIBLE); }
                    if do_l_v { lalpha[v][jp_v][dp] = lalpha[v][jp_v][dp].max(IMPOSSIBLE); }
                    if do_r_v { ralpha[v][jp_v][dp] = ralpha[v][jp_v][dp].max(IMPOSSIBLE); }
                }
            }
        } else if stt != B_ST {
            // D or S states (cm_dpalign_trunc.c:2519)
            for y in cfirst..(cfirst + cnum) {
                let do_j_y = cp9b.jvalid[y];
                let do_l_y = cp9b.lvalid[y] && fill_l;
                let do_r_y = cp9b.rvalid[y] && fill_r;
                let yoffset = (y - cfirst) as i32;
                let tsc = tsc_v[yoffset as usize];
                if (do_j_v && do_j_y) || (do_l_v && do_l_y) || (do_r_v && do_r_y) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = (jn - jmin[v]) as i32;
                    let jpx = (jx - jmin[v]) as i32;
                    let mut jp_y_sdr = (jn - jmin[y] - sdr) as i32;
                    while jp_v <= jpx {
                        let jp_v_u = jp_v as usize;
                        let jp_y_sdr_u = jp_y_sdr as usize;
                        let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_sdr_u] + sd);
                        let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_sdr_u] + sd);
                        let dpn = (dn - hdmin[v][jp_v_u]) as i32;
                        let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                        let mut dp_v = dpn;
                        let mut dp_y_sd = (dn - hdmin[y][jp_y_sdr_u] - sd) as i32;
                        while dp_v <= dpx {
                            let dp = dp_v as usize;
                            if do_j_v && do_j_y {
                                let sc = jalpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc;
                                if sc > jalpha[v][jp_v_u][dp] {
                                    jalpha[v][jp_v_u][dp] = sc;
                                    jysh[v][jp_v_u][dp] = yoffset + TRMODE_J_OFFSET;
                                }
                            }
                            if do_l_v && do_l_y {
                                let sc = lalpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc;
                                if sc > lalpha[v][jp_v_u][dp] {
                                    lalpha[v][jp_v_u][dp] = sc;
                                    lysh[v][jp_v_u][dp] = yoffset + TRMODE_L_OFFSET;
                                }
                            }
                            if do_r_v && do_r_y {
                                let sc = ralpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc;
                                if sc > ralpha[v][jp_v_u][dp] {
                                    ralpha[v][jp_v_u][dp] = sc;
                                    rysh[v][jp_v_u][dp] = yoffset + TRMODE_R_OFFSET;
                                }
                            }
                            // if d == 0, force L and R IMPOSSIBLE; reset S shadow (:2582)
                            if dp_v == dpn && dn == 0 {
                                if do_l_v { lalpha[v][jp_v_u][dp] = IMPOSSIBLE; }
                                if do_r_v { ralpha[v][jp_v_u][dp] = IMPOSSIBLE; }
                                if stt == S_ST {
                                    if do_l_v { lysh[v][jp_v_u][dp] = USED_TRUNC_END; }
                                    if do_r_v { rysh[v][jp_v_u][dp] = USED_TRUNC_END; }
                                }
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
            // B_st (cm_dpalign_trunc.c:2606)
            let y = cfirst; // left subtree
            let z = cnum; // right subtree
            let do_j_y = cp9b.jvalid[y];
            let do_l_y = cp9b.lvalid[y] && fill_l;
            let do_r_y = cp9b.rvalid[y] && fill_r;
            let do_j_z = cp9b.jvalid[z];
            let do_l_z = cp9b.lvalid[z] && fill_l;
            let do_r_z = cp9b.rvalid[z] && fill_r;

            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = (j - jmin[y]) as i32;
                let jp_z = (j - jmin[z]) as usize;
                let mut kn = (j - jmax[y]).max(hdmin[z][jp_z]);
                kn = kn.max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let row_y = (jp_y - k) as usize;
                        if k >= d - hdmax[y][row_y] && k <= d - hdmin[y][row_y] {
                            let kp_z = (k - hdmin[z][jp_z]) as usize;
                            let dp_y = d - hdmin[y][row_y];
                            let col_y = (dp_y - k) as usize;
                            if do_j_v && do_j_y && do_j_z {
                                let sc = jalpha[y][row_y][col_y] + jalpha[z][jp_z][kp_z];
                                if sc > jalpha[v][jp_v][dp_v] {
                                    jalpha[v][jp_v][dp_v] = sc;
                                    jksh[v][jp_v][dp_v] = k;
                                }
                            }
                            if do_l_v && do_j_y && do_l_z {
                                let sc = jalpha[y][row_y][col_y] + lalpha[z][jp_z][kp_z];
                                if sc > lalpha[v][jp_v][dp_v] {
                                    lalpha[v][jp_v][dp_v] = sc;
                                    lksh[v][jp_v][dp_v] = k;
                                    lkmode[v][jp_v][dp_v] = TRMODE_J;
                                }
                            }
                            if do_r_v && do_r_y && do_j_z {
                                let sc = ralpha[y][row_y][col_y] + jalpha[z][jp_z][kp_z];
                                if sc > ralpha[v][jp_v][dp_v] {
                                    ralpha[v][jp_v][dp_v] = sc;
                                    rksh[v][jp_v][dp_v] = k;
                                    rkmode[v][jp_v][dp_v] = TRMODE_J;
                                }
                            }
                            if k != 0 && k != d && do_t_v && do_r_y && do_l_z {
                                let sc = ralpha[y][row_y][col_y] + lalpha[z][jp_z][kp_z];
                                if sc > talpha[v][jp_v][dp_v] {
                                    talpha[v][jp_v][dp_v] = sc;
                                    tksh[v][jp_v][dp_v] = k;
                                }
                            }
                        }
                        k += 1;
                    }
                }
            }
            // special case: L, full seq on left, k==0 (cm_dpalign_trunc.c:2707)
            if do_l_v && (do_j_y || do_l_y) {
                let jn2 = jmin[v].max(jmin[y]);
                let jx2 = jmax[v].min(jmax[y]);
                for j in jn2..=jx2 {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_y = (j - jmin[y]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    for d in dn..=dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        if do_j_y {
                            let sc = jalpha[y][jp_y][dp_y];
                            if sc > lalpha[v][jp_v][dp_v] {
                                lalpha[v][jp_v][dp_v] = sc;
                                lksh[v][jp_v][dp_v] = 0;
                                lkmode[v][jp_v][dp_v] = TRMODE_J;
                            }
                        }
                        if do_l_y {
                            let sc = lalpha[y][jp_y][dp_y];
                            if sc > lalpha[v][jp_v][dp_v] {
                                lalpha[v][jp_v][dp_v] = sc;
                                lksh[v][jp_v][dp_v] = 0;
                                lkmode[v][jp_v][dp_v] = TRMODE_L;
                            }
                        }
                    }
                }
            }
            // special case: R, full seq on right, k==d (cm_dpalign_trunc.c:2739)
            if do_r_v && (do_j_z || do_r_z) {
                let jn2 = jmin[v].max(jmin[z]);
                let jx2 = jmax[v].min(jmax[z]);
                for j in jn2..=jx2 {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_z = (j - jmin[z]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[z][jp_z]);
                    let dx = hdmax[v][jp_v].min(hdmax[z][jp_z]);
                    for d in dn..=dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_z = (d - hdmin[z][jp_z]) as usize;
                        if do_j_z {
                            let sc = jalpha[z][jp_z][dp_z];
                            if sc > ralpha[v][jp_v][dp_v] {
                                ralpha[v][jp_v][dp_v] = sc;
                                rksh[v][jp_v][dp_v] = d;
                                rkmode[v][jp_v][dp_v] = TRMODE_J;
                            }
                        }
                        if do_r_z {
                            let sc = ralpha[z][jp_z][dp_z];
                            if sc > ralpha[v][jp_v][dp_v] {
                                ralpha[v][jp_v][dp_v] = sc;
                                rksh[v][jp_v][dp_v] = d;
                                rkmode[v][jp_v][dp_v] = TRMODE_R;
                            }
                        }
                    }
                }
            }
        }

        // ---- ROOT_S truncated-begin update (cm_dpalign_trunc.c:2775) ----
        if l >= jmin[v] && l <= jmax[v] {
            let jp_v = (l - jmin[v]) as usize;
            if l >= hdmin[v][jp_v] && l <= hdmax[v][jp_v] {
                let lpv = (l - hdmin[v][jp_v]) as usize;
                let trpenalty = pty[v];
                if not_impossible(trpenalty) {
                    if do_j_v && cp9b.jvalid[0] {
                        let sc = jalpha[v][jp_v][lpv] + trpenalty;
                        if sc > jalpha[0][jp_0][lp_0] {
                            jalpha[0][jp_0][lp_0] = sc;
                            jb = v as i32;
                        }
                    }
                    if do_l_v && cp9b.lvalid[0] {
                        let sc = lalpha[v][jp_v][lpv] + trpenalty;
                        if sc > lalpha[0][jp_0][lp_0] {
                            lalpha[0][jp_0][lp_0] = sc;
                            lb = v as i32;
                        }
                    }
                    if do_r_v && cp9b.rvalid[0] {
                        let sc = ralpha[v][jp_v][lpv] + trpenalty;
                        if sc > ralpha[0][jp_0][lp_0] {
                            ralpha[0][jp_0][lp_0] = sc;
                            rb = v as i32;
                        }
                    }
                    if do_t_v && cp9b.tvalid[0] {
                        let sc = talpha[v][jp_v][lpv] + trpenalty;
                        if sc > talpha[0][jp_0][lp_0] {
                            talpha[0][jp_0][lp_0] = sc;
                            tb = v as i32;
                        }
                    }
                }
            }
        }
    } // end for v

    // all valid alignments use a truncated begin (cm_dpalign_trunc.c:2818)
    if cp9b.jvalid[0] { jysh[0][jp_0][lp_0] = USED_TRUNC_BEGIN; }
    if fill_l && cp9b.lvalid[0] { lysh[0][jp_0][lp_0] = USED_TRUNC_BEGIN; }
    if fill_r && cp9b.rvalid[0] { rysh[0][jp_0][lp_0] = USED_TRUNC_BEGIN; }

    // determine mode (cm_dpalign_trunc.c:2825)
    let (mut sc, mut mode, mut b);
    match preset_mode {
        TRMODE_J => { sc = jalpha[0][jp_0][lp_0]; mode = TRMODE_J; b = jb; }
        TRMODE_L => { sc = lalpha[0][jp_0][lp_0]; mode = TRMODE_L; b = lb; }
        TRMODE_R => { sc = ralpha[0][jp_0][lp_0]; mode = TRMODE_R; b = rb; }
        TRMODE_T => { sc = talpha[0][jp_0][lp_0]; mode = TRMODE_T; b = tb; }
        _ => {
            sc = IMPOSSIBLE;
            mode = TRMODE_UNKNOWN;
            b = 0;
            if cp9b.jvalid[0] && jalpha[0][jp_0][lp_0] > sc {
                sc = jalpha[0][jp_0][lp_0]; mode = TRMODE_J; b = jb;
            }
            if fill_l && cp9b.lvalid[0] && lalpha[0][jp_0][lp_0] > sc {
                sc = lalpha[0][jp_0][lp_0]; mode = TRMODE_L; b = lb;
            }
            if fill_r && cp9b.rvalid[0] && ralpha[0][jp_0][lp_0] > sc {
                sc = ralpha[0][jp_0][lp_0]; mode = TRMODE_R; b = rb;
            }
            if fill_t && cp9b.tvalid[0] && talpha[0][jp_0][lp_0] > sc {
                sc = talpha[0][jp_0][lp_0]; mode = TRMODE_T; b = tb;
            }
        }
    }
    if mode == TRMODE_UNKNOWN || !not_impossible(sc) {
        // C would ESL_FAIL; pli_align_hit fails over. Use non-banded TrCYK for bounds.
        return tr_cyk_align(cm, trp, lm, rm, dsq, lp, pass_idx, use_local);
    }

    // ---- traceback: CYK branch of cm_tr_alignT_hb (cm_dpalign_trunc.c:471) ----
    let opt_mode = mode;
    let mut tr = Parsetree::new(lp as usize + 4);
    tr.is_std = false; // C cm_tr_alignT_hb: truncated parse
    tr.pass_idx = pass_idx;
    // C cm_dpalign_trunc.c:540: tr->trpenalty = (local?l:g)_ptyAA[pty_idx][b].
    tr.trpenalty = pty[b as usize];
    tr.add_node_mode(1, l, 0, -1, -1, -1, mode); // attach root ROOT_S
    let mut pda_i: Vec<i32> = Vec::new();
    let mut pda_c: Vec<i8> = Vec::new();
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = l;
    let mut d: i32 = l;

    loop {
        let vu = v as usize;
        // super-special case: BEGL_S/BEGR_S, d==0, mode/band-disallowed (:552)
        let mut allow_s_trunc_end = false;
        let allow_s_local_end = false; // only set if do_optacc (FALSE here)
        let mut jp_v = 0usize;
        let mut dp_v = 0usize;
        if v != cm.m && cm.sttype[vu] as i32 != E_ST {
            let stid = cm.stid[vu] as i32;
            let mode_disallowed = (mode == TRMODE_J && !cp9b.jvalid[vu])
                || (mode == TRMODE_L && !cp9b.lvalid[vu])
                || (mode == TRMODE_R && !cp9b.rvalid[vu]);
            let out_of_band = j < jmin[vu]
                || j > jmax[vu]
                || {
                    let jpv = j - jmin[vu];
                    jpv < 0 || d < hdmin[vu][jpv as usize] || d > hdmax[vu][jpv as usize]
                };
            if (stid == BEGL_S || stid == BEGR_S)
                && d == 0
                && (mode_disallowed || out_of_band)
            {
                if (stid == BEGL_S && mode == TRMODE_R)
                    || (stid == BEGR_S && mode == TRMODE_L)
                {
                    allow_s_trunc_end = true;
                }
                // else if do_optacc { allow_s_local_end = true } — do_optacc FALSE
            } else if cm.sttype[vu] as i32 != EL_ST {
                jp_v = (j - jmin[vu]) as usize;
                dp_v = (d - hdmin[vu][jp_v]) as usize;
            }
        }

        if v != cm.m && cm.sttype[vu] as i32 == B_ST {
            let k = match mode {
                TRMODE_J => jksh[vu][jp_v][dp_v],
                TRMODE_L => lksh[vu][jp_v][dp_v],
                TRMODE_R => rksh[vu][jp_v][dp_v],
                _ => tksh[vu][jp_v][dp_v],
            };
            let prvmode = mode;
            let rmode = match mode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => TRMODE_L,
                TRMODE_R => rkmode[vu][jp_v][dp_v],
                _ => TRMODE_L, // T
            };
            let bpar = tr.n - 1;
            pda_c.push(rmode);
            pda_i.push(j);
            pda_i.push(k);
            pda_i.push(bpar);
            let lmode = match prvmode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => lkmode[vu][jp_v][dp_v],
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
        } else if v == cm.m || cm.sttype[vu] as i32 == E_ST || cm.sttype[vu] as i32 == EL_ST {
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
        } else {
            // get yoffset (:665)
            let yoffset_raw = if allow_s_trunc_end {
                USED_TRUNC_END
            } else if allow_s_local_end {
                USED_EL
            } else {
                match mode {
                    TRMODE_J => jysh[vu][jp_v][dp_v],
                    TRMODE_L => lysh[vu][jp_v][dp_v],
                    TRMODE_R => rysh[vu][jp_v][dp_v],
                    _ => {
                        // TRMODE_T: only valid at v==0 (USED_TRUNC_BEGIN)
                        if v == 0 { USED_TRUNC_BEGIN } else { USED_TRUNC_BEGIN }
                    }
                }
            };
            let mut nxtmode = mode;
            let yoffset;
            if yoffset_raw == USED_TRUNC_BEGIN {
                nxtmode = mode;
                yoffset = USED_TRUNC_BEGIN;
            } else if yoffset_raw == USED_TRUNC_END {
                yoffset = USED_TRUNC_END;
            } else if yoffset_raw == USED_EL {
                yoffset = USED_EL;
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

            let stt = cm.sttype[vu] as i32;
            match stt {
                x if x == MP_ST => {
                    if mode == TRMODE_J { i += 1; }
                    if mode == TRMODE_L && d > 0 { i += 1; }
                    if mode == TRMODE_J { j -= 1; }
                    if mode == TRMODE_R && d > 0 { j -= 1; }
                }
                x if x == ML_ST || x == IL_ST => {
                    if mode == TRMODE_J { i += 1; }
                    if mode == TRMODE_L && d > 0 { i += 1; }
                }
                x if x == MR_ST || x == IR_ST => {
                    if mode == TRMODE_J { j -= 1; }
                    if mode == TRMODE_R && d > 0 { j -= 1; }
                }
                _ => {} // D, S
            }
            d = j - i + 1;

            if yoffset == USED_EL || yoffset == USED_TRUNC_END {
                if yoffset == USED_EL {
                    let idx = tr.add_node_mode(i, j, cm.m, -1, -1, tr.n - 1, mode);
                    tr.nxtl[(tr.n - 2) as usize] = idx;
                }
                v = cm.m;
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
    }

    (tr, sc, opt_mode)
}

/// HMM-banded truncated Inside matrix (FLogsum). Decks are `[v][jp][dp]` with the
/// same band offsets as [`tr_cyk_align_hb`]; index `v == cm.M` is the (non-banded)
/// EL deck. `t` is allocated for B states and the ROOT_S (v==0) only.
pub struct TrHbMx {
    pub j: Vec<Vec<Vec<f32>>>,
    pub l: Vec<Vec<Vec<f32>>>,
    pub r: Vec<Vec<Vec<f32>>>,
    pub t: Vec<Vec<Vec<f32>>>,
}

/// C `cm_TrInsideAlignHB` (cm_dpalign_trunc.c:3431): HMM-banded truncated Inside DP
/// (FLogsum). Returns the filled matrix, the determined `mode`, and the Inside score
/// at ROOT_S [L][L]. Structural twin of [`tr_cyk_align_hb`] with max/shadow replaced
/// by FLogsum and no traceback decks; adds the EL deck (v==cm.M) + per-state local-end
/// (endsc) re-init needed for the LOCAL (cmalign default) config.
#[allow(clippy::too_many_arguments)]
pub fn tr_inside_align_hb(
    cm: &CM,
    trp: &TrPenalties,
    cp9b: &crate::cp9::CP9Bands,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    lp: i32,
    pass_idx: i32,
    preset_mode: i8,
    use_local: bool,
) -> (TrHbMx, i8, f32) {
    let m = cm.m as usize;
    let l = lp;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let kp = crate::cm::ALPHABET_SIZE_P;

    let (fill_l, fill_r, fill_t) = match preset_mode {
        TRMODE_J => (false, false, false),
        TRMODE_L => (true, false, false),
        TRMODE_R => (false, true, false),
        _ => (true, true, true),
    };
    let pty = trp.pty_slice(pass_idx, use_local);
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    // el_scA[d] = el_selfsc * d (C 3513).
    let mut el_sca = vec![0.0f32; (l + 1) as usize];
    for d in 0..=l as usize {
        el_sca[d] = cm.el_selfsc * d as f32;
    }

    // ---- allocate banded {J,L,R,T} decks; index m is the non-banded EL deck ----
    let njv = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut jalpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut lalpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut ralpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut talpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    for v in 0..m {
        let nj = njv(v);
        let is_b = cm.sttype[v] as i32 == B_ST;
        let mut jd = Vec::with_capacity(nj);
        let mut ld = Vec::with_capacity(nj);
        let mut rd = Vec::with_capacity(nj);
        let mut td = Vec::with_capacity(nj);
        for jp in 0..nj {
            let w = (hdmax[v][jp] - hdmin[v][jp] + 1).max(0) as usize;
            jd.push(vec![IMPOSSIBLE; w]);
            ld.push(vec![IMPOSSIBLE; w]);
            rd.push(vec![IMPOSSIBLE; w]);
            if is_b || v == 0 {
                td.push(vec![IMPOSSIBLE; w]);
            }
        }
        jalpha.push(jd);
        lalpha.push(ld);
        ralpha.push(rd);
        talpha.push(td);
    }
    // EL deck (v == cm.M): NON-banded triangular (C cm_mx.c: EL deck is non-banded),
    // rows j=0..=L each of width j+1 (d=0..=j). Allocated only if the marginal plane
    // is valid for state M (C checks cp9b->{J,L,R}valid[cm->M]).
    let lu = l as usize;
    let el_rows = |valid: bool| -> Vec<Vec<f32>> {
        if valid {
            (0..=lu).map(|j| vec![IMPOSSIBLE; j + 1]).collect()
        } else {
            Vec::new()
        }
    };
    jalpha.push(el_rows(cp9b.jvalid[m]));
    lalpha.push(el_rows(fill_l && cp9b.lvalid[m]));
    ralpha.push(el_rows(fill_r && cp9b.rvalid[m]));
    talpha.push(Vec::new()); // no T EL deck

    // fill EL deck with el_scA (C 3535-3557) if local ends on.
    if local_end {
        if cp9b.jvalid[m] {
            for j in 0..=lu {
                for d in 0..=j {
                    jalpha[m][j][d] = el_sca[d];
                }
            }
        }
        if fill_l && cp9b.lvalid[m] {
            for j in 0..=lu {
                for d in 0..=j {
                    lalpha[m][j][d] = el_sca[d];
                }
            }
        }
        if fill_r && cp9b.rvalid[m] {
            for j in 0..=lu {
                for d in 0..=j {
                    ralpha[m][j][d] = el_sca[d];
                }
            }
        }
    }

    // ---- main recursion v = M-1 downto 1 (C 3560) ----
    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = tr_state_delta(stt);
        let sdl = tr_state_left_delta(stt);
        let sdr = tr_state_right_delta(stt);
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];
        let lmesc_v = &lm[v];
        let rmesc_v = &rm[v];
        let endsc_v = cm.endsc[v];
        let do_j_v = cp9b.jvalid[v];
        let do_l_v = cp9b.lvalid[v] && fill_l;
        let do_r_v = cp9b.rvalid[v] && fill_r;
        let do_t_v = cp9b.tvalid[v] && fill_t;

        // re-init J/L/R decks for a local end from v (C 3570-3612), via el_scA.
        if not_impossible(endsc_v) {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                if do_j_v && cp9b.jvalid[m] {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sd {
                        (hdmin[v][jp_v], 0i32)
                    } else {
                        (sd, sd - hdmin[v][jp_v])
                    };
                    while d <= hdmax[v][jp_v] {
                        jalpha[v][jp_v][dp_v as usize] = el_sca[(d - sd) as usize] + endsc_v;
                        d += 1;
                        dp_v += 1;
                    }
                }
                if do_l_v && cp9b.lvalid[m] {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdl {
                        (hdmin[v][jp_v], 0i32)
                    } else {
                        (sdl, sdl - hdmin[v][jp_v])
                    };
                    while d <= hdmax[v][jp_v] {
                        lalpha[v][jp_v][dp_v as usize] = el_sca[(d - sdl) as usize] + endsc_v;
                        d += 1;
                        dp_v += 1;
                    }
                }
                if do_r_v && cp9b.rvalid[m] {
                    let (mut d, mut dp_v) = if hdmin[v][jp_v] >= sdr {
                        (hdmin[v][jp_v], 0i32)
                    } else {
                        (sdr, sdr - hdmin[v][jp_v])
                    };
                    while d <= hdmax[v][jp_v] {
                        ralpha[v][jp_v][dp_v as usize] = el_sca[(d - sdr) as usize] + endsc_v;
                        d += 1;
                        dp_v += 1;
                    }
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                if do_j_v { jalpha[v][jp_v][0] = 0.0; }
                if do_l_v { lalpha[v][jp_v][0] = 0.0; }
                if do_r_v { ralpha[v][jp_v][0] = 0.0; }
            }
        } else if stt == IL_ST || stt == ML_ST {
            if !state_is_detached(cm, v) {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let j_sdr = j - sdr;
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        if j_sdr >= jmin[y] && j_sdr <= jmax[y] { yvalid.push(yoffset); }
                    }
                    for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                        let i = j - d + 1;
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        // J and L (use j_sdr, d_sd)
                        if do_j_v || do_l_v {
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_j_y = cp9b.jvalid[y];
                                let do_l_y = cp9b.lvalid[y] && fill_l;
                                if do_j_y || do_l_y {
                                    let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                    if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                        let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                        if do_j_v && do_j_y {
                                            jalpha[v][jp_v][dp_v] = flogsum(jalpha[v][jp_v][dp_v], jalpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset]);
                                        }
                                        if do_l_v && do_l_y {
                                            lalpha[v][jp_v][dp_v] = flogsum(lalpha[v][jp_v][dp_v], lalpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset]);
                                        }
                                    }
                                }
                            }
                            if do_j_v {
                                jalpha[v][jp_v][dp_v] += esc_v[dsq[i as usize] as usize];
                                jalpha[v][jp_v][dp_v] = jalpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                            if do_l_v {
                                lalpha[v][jp_v][dp_v] = if d >= 2 {
                                    lalpha[v][jp_v][dp_v] + esc_v[dsq[i as usize] as usize]
                                } else {
                                    esc_v[dsq[i as usize] as usize]
                                };
                                lalpha[v][jp_v][dp_v] = lalpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                        }
                        // R (use j_sdr, d)
                        if do_r_v {
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_r_y = cp9b.rvalid[y] && fill_r;
                                let do_j_y = cp9b.jvalid[y];
                                if (do_j_y || do_r_y) && y != v {
                                    let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                    if d >= hdmin[y][jp_y_sdr] && d <= hdmax[y][jp_y_sdr] {
                                        let dp_y = (d - hdmin[y][jp_y_sdr]) as usize;
                                        if do_j_y {
                                            ralpha[v][jp_v][dp_v] = flogsum(ralpha[v][jp_v][dp_v], jalpha[y][jp_y_sdr][dp_y] + tsc_v[yoffset]);
                                        }
                                        if do_r_y {
                                            ralpha[v][jp_v][dp_v] = flogsum(ralpha[v][jp_v][dp_v], ralpha[y][jp_y_sdr][dp_y] + tsc_v[yoffset]);
                                        }
                                    }
                                }
                            }
                            ralpha[v][jp_v][dp_v] = ralpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                        }
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            if !state_is_detached(cm, v) {
                if do_j_v || do_r_v {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let j_sdr = j - sdr;
                        let mut yvalid: Vec<usize> = Vec::new();
                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            if j_sdr >= jmin[y] && j_sdr <= jmax[y] { yvalid.push(yoffset); }
                        }
                        for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                            let dp_v = (d - hdmin[v][jp_v]) as usize;
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_j_y = cp9b.jvalid[y];
                                let do_r_y = cp9b.rvalid[y] && fill_r;
                                if do_j_y || do_r_y {
                                    let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                    if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                                        let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                                        if do_j_v && do_j_y {
                                            jalpha[v][jp_v][dp_v] = flogsum(jalpha[v][jp_v][dp_v], jalpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset]);
                                        }
                                        if do_r_v && do_r_y {
                                            ralpha[v][jp_v][dp_v] = flogsum(ralpha[v][jp_v][dp_v], ralpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset]);
                                        }
                                    }
                                }
                            }
                            if do_j_v {
                                jalpha[v][jp_v][dp_v] += esc_v[dsq[j as usize] as usize];
                                jalpha[v][jp_v][dp_v] = jalpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                            if do_r_v {
                                ralpha[v][jp_v][dp_v] = if d >= 2 {
                                    ralpha[v][jp_v][dp_v] + esc_v[dsq[j as usize] as usize]
                                } else {
                                    esc_v[dsq[j as usize] as usize]
                                };
                                ralpha[v][jp_v][dp_v] = ralpha[v][jp_v][dp_v].max(IMPOSSIBLE);
                            }
                        }
                    }
                }
                if do_l_v {
                    for j in jmin[v]..=jmax[v] {
                        let jp_v = (j - jmin[v]) as usize;
                        let mut yvalid: Vec<usize> = Vec::new();
                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            if j >= jmin[y] && j <= jmax[y] { yvalid.push(yoffset); }
                        }
                        for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                            let dp_v = (d - hdmin[v][jp_v]) as usize;
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let do_l_y = cp9b.lvalid[y] && fill_l;
                                let do_j_y = cp9b.jvalid[y];
                                if (do_j_y || do_l_y) && y != v {
                                    let jp_y = (j - jmin[y]) as usize;
                                    if d >= hdmin[y][jp_y] && d <= hdmax[y][jp_y] {
                                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                                        if do_j_y {
                                            lalpha[v][jp_v][dp_v] = flogsum(lalpha[v][jp_v][dp_v], jalpha[y][jp_y][dp_y] + tsc_v[yoffset]);
                                        }
                                        if do_l_y {
                                            lalpha[v][jp_v][dp_v] = flogsum(lalpha[v][jp_v][dp_v], lalpha[y][jp_y][dp_y] + tsc_v[yoffset]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if stt == MP_ST {
            for y in cfirst..(cfirst + cnum) {
                let do_j_y = cp9b.jvalid[y];
                let do_l_y = cp9b.lvalid[y] && fill_l;
                let do_r_y = cp9b.rvalid[y] && fill_r;
                let yoffset = (y - cfirst) as i32;
                let tsc = tsc_v[yoffset as usize];
                // J and R (j set: j-sdr in y's band)
                if (do_j_v && do_j_y) || (do_r_v && (do_j_y || do_r_y)) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = (jn - jmin[v]) as i32;
                    let jpx = (jx - jmin[v]) as i32;
                    let mut jp_y_sdr = (jn - jmin[y] - sdr) as i32;
                    while jp_v <= jpx {
                        let jp_v_u = jp_v as usize;
                        let jp_y_sdr_u = jp_y_sdr as usize;
                        if do_j_v && do_j_y {
                            let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_sdr_u] + sd);
                            let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_sdr_u] + sd);
                            let mut dp_v = (dn - hdmin[v][jp_v_u]) as i32;
                            let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                            let mut dp_y_sd = (dn - hdmin[y][jp_y_sdr_u] - sd) as i32;
                            while dp_v <= dpx {
                                jalpha[v][jp_v_u][dp_v as usize] = flogsum(jalpha[v][jp_v_u][dp_v as usize], jalpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc);
                                dp_v += 1;
                                dp_y_sd += 1;
                            }
                        }
                        if do_r_v && (do_r_y || do_j_y) {
                            let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_sdr_u] + sdr);
                            let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_sdr_u] + sdr);
                            let mut dp_v = (dn - hdmin[v][jp_v_u]) as i32;
                            let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                            let mut dp_y_sdr = (dn - hdmin[y][jp_y_sdr_u] - sdr) as i32;
                            while dp_v <= dpx {
                                if do_j_y {
                                    ralpha[v][jp_v_u][dp_v as usize] = flogsum(ralpha[v][jp_v_u][dp_v as usize], jalpha[y][jp_y_sdr_u][dp_y_sdr as usize] + tsc);
                                }
                                if do_r_y {
                                    ralpha[v][jp_v_u][dp_v as usize] = flogsum(ralpha[v][jp_v_u][dp_v as usize], ralpha[y][jp_y_sdr_u][dp_y_sdr as usize] + tsc);
                                }
                                dp_v += 1;
                                dp_y_sdr += 1;
                            }
                        }
                        jp_v += 1;
                        jp_y_sdr += 1;
                    }
                }
                // L (j set: j in y's band)
                if do_l_v && (do_l_y || do_j_y) {
                    let jn = jmin[v].max(jmin[y]);
                    let jx = jmax[v].min(jmax[y]);
                    let mut jp_v = (jn - jmin[v]) as i32;
                    let jpx = (jx - jmin[v]) as i32;
                    let mut jp_y = (jn - jmin[y]) as i32;
                    while jp_v <= jpx {
                        let jp_v_u = jp_v as usize;
                        let jp_y_u = jp_y as usize;
                        let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_u] + sdl);
                        let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_u] + sdl);
                        let mut dp_v = (dn - hdmin[v][jp_v_u]) as i32;
                        let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                        let mut dp_y_sdl = (dn - hdmin[y][jp_y_u] - sdl) as i32;
                        while dp_v <= dpx {
                            if do_j_y {
                                lalpha[v][jp_v_u][dp_v as usize] = flogsum(lalpha[v][jp_v_u][dp_v as usize], jalpha[y][jp_y_u][dp_y_sdl as usize] + tsc);
                            }
                            if do_l_y {
                                lalpha[v][jp_v_u][dp_v as usize] = flogsum(lalpha[v][jp_v_u][dp_v as usize], lalpha[y][jp_y_u][dp_y_sdl as usize] + tsc);
                            }
                            dp_v += 1;
                            dp_y_sdl += 1;
                        }
                        jp_v += 1;
                        jp_y += 1;
                    }
                }
            }
            // MP emissions (C 3218-3231)
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut i = j - hdmin[v][jp_v] + 1;
                let mut dp_v = 0usize;
                for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                    if d >= 2 {
                        if do_j_v {
                            let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                            jalpha[v][jp_v][dp_v] += esc_v[idx];
                        }
                        if do_l_v { lalpha[v][jp_v][dp_v] += lmesc_v[dsq[i as usize] as usize]; }
                        if do_r_v { ralpha[v][jp_v][dp_v] += rmesc_v[dsq[j as usize] as usize]; }
                    } else {
                        if do_j_v { jalpha[v][jp_v][dp_v] = IMPOSSIBLE; }
                        if do_l_v { lalpha[v][jp_v][dp_v] = lmesc_v[dsq[i as usize] as usize]; }
                        if do_r_v { ralpha[v][jp_v][dp_v] = rmesc_v[dsq[j as usize] as usize]; }
                    }
                    i -= 1;
                    dp_v += 1;
                }
            }
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let ww = hdmax[v][jp_v] - hdmin[v][jp_v];
                for dp in 0..=ww.max(-1) {
                    if ww < 0 { break; }
                    let dp = dp as usize;
                    if do_j_v { jalpha[v][jp_v][dp] = jalpha[v][jp_v][dp].max(IMPOSSIBLE); }
                    if do_l_v { lalpha[v][jp_v][dp] = lalpha[v][jp_v][dp].max(IMPOSSIBLE); }
                    if do_r_v { ralpha[v][jp_v][dp] = ralpha[v][jp_v][dp].max(IMPOSSIBLE); }
                }
            }
        } else if stt != B_ST {
            // D, S states (C 3241-3267)
            for y in cfirst..(cfirst + cnum) {
                let do_j_y = cp9b.jvalid[y];
                let do_l_y = cp9b.lvalid[y] && fill_l;
                let do_r_y = cp9b.rvalid[y] && fill_r;
                let yoffset = (y - cfirst) as i32;
                let tsc = tsc_v[yoffset as usize];
                if (do_j_v && do_j_y) || (do_l_v && do_l_y) || (do_r_v && do_r_y) {
                    let jn = jmin[v].max(jmin[y] + sdr);
                    let jx = jmax[v].min(jmax[y] + sdr);
                    let mut jp_v = (jn - jmin[v]) as i32;
                    let jpx = (jx - jmin[v]) as i32;
                    let mut jp_y_sdr = (jn - jmin[y] - sdr) as i32;
                    while jp_v <= jpx {
                        let jp_v_u = jp_v as usize;
                        let jp_y_sdr_u = jp_y_sdr as usize;
                        let dn = hdmin[v][jp_v_u].max(hdmin[y][jp_y_sdr_u] + sd);
                        let dx = hdmax[v][jp_v_u].min(hdmax[y][jp_y_sdr_u] + sd);
                        let dpn = (dn - hdmin[v][jp_v_u]) as i32;
                        let dpx = (dx - hdmin[v][jp_v_u]) as i32;
                        let mut dp_v = dpn;
                        let mut dp_y_sd = (dn - hdmin[y][jp_y_sdr_u] - sd) as i32;
                        while dp_v <= dpx {
                            let dp = dp_v as usize;
                            if do_j_v && do_j_y {
                                jalpha[v][jp_v_u][dp] = flogsum(jalpha[v][jp_v_u][dp], jalpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc);
                            }
                            if do_l_v && do_l_y {
                                lalpha[v][jp_v_u][dp] = flogsum(lalpha[v][jp_v_u][dp], lalpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc);
                            }
                            if do_r_v && do_r_y {
                                ralpha[v][jp_v_u][dp] = flogsum(ralpha[v][jp_v_u][dp], ralpha[y][jp_y_sdr_u][dp_y_sd as usize] + tsc);
                            }
                            if dp_v == dpn && dn == 0 {
                                if do_l_v { lalpha[v][jp_v_u][dp] = IMPOSSIBLE; }
                                if do_r_v { ralpha[v][jp_v_u][dp] = IMPOSSIBLE; }
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
            // B_st (C 3268-3298)
            let y = cfirst;
            let z = cnum;
            let do_j_y = cp9b.jvalid[y];
            let do_r_y = cp9b.rvalid[y] && fill_r;
            let do_j_z = cp9b.jvalid[z];
            let do_l_z = cp9b.lvalid[z] && fill_l;
            let do_r_z = cp9b.rvalid[z] && fill_r;

            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = (j - jmin[y]) as i32;
                let jp_z = (j - jmin[z]) as usize;
                let mut kn = (j - jmax[y]).max(hdmin[z][jp_z]);
                kn = kn.max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                for d in hdmin[v][jp_v]..=hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let row_y = (jp_y - k) as usize;
                        if k >= d - hdmax[y][row_y] && k <= d - hdmin[y][row_y] {
                            let kp_z = (k - hdmin[z][jp_z]) as usize;
                            let dp_y = d - hdmin[y][row_y];
                            let col_y = (dp_y - k) as usize;
                            if do_j_v && do_j_y && do_j_z {
                                jalpha[v][jp_v][dp_v] = flogsum(jalpha[v][jp_v][dp_v], jalpha[y][row_y][col_y] + jalpha[z][jp_z][kp_z]);
                            }
                            if do_l_v && do_j_y && do_l_z {
                                lalpha[v][jp_v][dp_v] = flogsum(lalpha[v][jp_v][dp_v], jalpha[y][row_y][col_y] + lalpha[z][jp_z][kp_z]);
                            }
                            if do_r_v && do_r_y && do_j_z {
                                ralpha[v][jp_v][dp_v] = flogsum(ralpha[v][jp_v][dp_v], ralpha[y][row_y][col_y] + jalpha[z][jp_z][kp_z]);
                            }
                            if k != 0 && k != d && do_t_v && do_r_y && do_l_z {
                                talpha[v][jp_v][dp_v] = flogsum(talpha[v][jp_v][dp_v], ralpha[y][row_y][col_y] + lalpha[z][jp_z][kp_z]);
                            }
                        }
                        k += 1;
                    }
                }
            }
            // L, full seq on left, k==0 (C 2707)
            if do_l_v && (do_j_y || (cp9b.lvalid[y] && fill_l)) {
                let jn2 = jmin[v].max(jmin[y]);
                let jx2 = jmax[v].min(jmax[y]);
                for j in jn2..=jx2 {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_y = (j - jmin[y]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    for d in dn..=dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        if do_j_y {
                            lalpha[v][jp_v][dp_v] = flogsum(lalpha[v][jp_v][dp_v], jalpha[y][jp_y][dp_y]);
                        }
                        if cp9b.lvalid[y] && fill_l {
                            lalpha[v][jp_v][dp_v] = flogsum(lalpha[v][jp_v][dp_v], lalpha[y][jp_y][dp_y]);
                        }
                    }
                }
            }
            // R, full seq on right, k==d (C 2739)
            if do_r_v && (do_j_z || do_r_z) {
                let jn2 = jmin[v].max(jmin[z]);
                let jx2 = jmax[v].min(jmax[z]);
                for j in jn2..=jx2 {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_z = (j - jmin[z]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[z][jp_z]);
                    let dx = hdmax[v][jp_v].min(hdmax[z][jp_z]);
                    for d in dn..=dx {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_z = (d - hdmin[z][jp_z]) as usize;
                        if do_j_z {
                            ralpha[v][jp_v][dp_v] = flogsum(ralpha[v][jp_v][dp_v], jalpha[z][jp_z][dp_z]);
                        }
                        if do_r_z {
                            ralpha[v][jp_v][dp_v] = flogsum(ralpha[v][jp_v][dp_v], ralpha[z][jp_z][dp_z]);
                        }
                    }
                }
            }
        }

        // ---- ROOT_S truncated-begin update (C 3300-3328) ----
        if l >= jmin[v] && l <= jmax[v] {
            let jp_v = (l - jmin[v]) as usize;
            if l >= hdmin[v][jp_v] && l <= hdmax[v][jp_v] {
                let lpv = (l - hdmin[v][jp_v]) as usize;
                let trpenalty = pty[v];
                if not_impossible(trpenalty) {
                    if do_j_v && cp9b.jvalid[0] {
                        jalpha[0][jp_0][lp_0] = flogsum(jalpha[0][jp_0][lp_0], jalpha[v][jp_v][lpv] + trpenalty);
                    }
                    if do_l_v && cp9b.lvalid[0] {
                        lalpha[0][jp_0][lp_0] = flogsum(lalpha[0][jp_0][lp_0], lalpha[v][jp_v][lpv] + trpenalty);
                    }
                    if do_r_v && cp9b.rvalid[0] {
                        ralpha[0][jp_0][lp_0] = flogsum(ralpha[0][jp_0][lp_0], ralpha[v][jp_v][lpv] + trpenalty);
                    }
                    if do_t_v && cp9b.tvalid[0] {
                        talpha[0][jp_0][lp_0] = flogsum(talpha[0][jp_0][lp_0], talpha[v][jp_v][lpv] + trpenalty);
                    }
                }
            }
        }
    } // end for v

    // ---- determine mode of optimal alignment (C 3331-3363) ----
    let (mut sc, mut mode);
    match preset_mode {
        TRMODE_J => { sc = jalpha[0][jp_0][lp_0]; mode = TRMODE_J; }
        TRMODE_L => { sc = lalpha[0][jp_0][lp_0]; mode = TRMODE_L; }
        TRMODE_R => { sc = ralpha[0][jp_0][lp_0]; mode = TRMODE_R; }
        TRMODE_T => { sc = talpha[0][jp_0][lp_0]; mode = TRMODE_T; }
        _ => {
            sc = jalpha[0][jp_0][lp_0]; mode = TRMODE_J;
            if fill_l && lalpha[0][jp_0][lp_0] > sc { sc = lalpha[0][jp_0][lp_0]; mode = TRMODE_L; }
            if fill_r && ralpha[0][jp_0][lp_0] > sc { sc = ralpha[0][jp_0][lp_0]; mode = TRMODE_R; }
            if fill_t && talpha[0][jp_0][lp_0] > sc { sc = talpha[0][jp_0][lp_0]; mode = TRMODE_T; }
        }
    }
    let _ = &mut sc;
    let _ = &mut mode;
    let mx = TrHbMx { j: jalpha, l: lalpha, r: ralpha, t: talpha };
    (mx, mode, sc)
}

/// C `cm_TrOutsideAlignHB` (cm_dpalign_trunc.c:7907): HMM-banded truncated Outside DP
/// (FLogsum). `preset_mode` is the determined optimal mode (J/L/R/T). Consumes the
/// filled Inside matrix `ins`; returns the beta matrix. do_check not ported.
#[allow(clippy::too_many_arguments)]
pub fn tr_outside_align_hb(
    cm: &CM,
    trp: &TrPenalties,
    cp9b: &crate::cp9::CP9Bands,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    lp: i32,
    preset_mode: i8,
    pass_idx: i32,
    use_local: bool,
    ins: &TrHbMx,
) -> TrHbMx {
    let m = cm.m as usize;
    let l = lp;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let kp = crate::cm::ALPHABET_SIZE_P;
    let (fill_l, fill_r, fill_t) = tr_fill_from_mode(preset_mode);
    let pty = trp.pty_slice(pass_idx, use_local);
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    // ---- allocate beta decks (same shape as the Inside matrix) ----
    let njv = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut jbeta: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut lbeta: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut rbeta: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut tbeta: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    for v in 0..m {
        let nj = njv(v);
        let is_b = cm.sttype[v] as i32 == B_ST;
        let mut jd = Vec::with_capacity(nj);
        let mut ld = Vec::with_capacity(nj);
        let mut rd = Vec::with_capacity(nj);
        let mut td = Vec::with_capacity(nj);
        for jp in 0..nj {
            let w = (hdmax[v][jp] - hdmin[v][jp] + 1).max(0) as usize;
            jd.push(vec![IMPOSSIBLE; w]);
            ld.push(vec![IMPOSSIBLE; w]);
            rd.push(vec![IMPOSSIBLE; w]);
            if is_b || v == 0 {
                td.push(vec![IMPOSSIBLE; w]);
            }
        }
        jbeta.push(jd);
        lbeta.push(ld);
        rbeta.push(rd);
        tbeta.push(td);
    }
    let lu = l as usize;
    let el_rows = |valid: bool| -> Vec<Vec<f32>> {
        if valid { (0..=lu).map(|j| vec![IMPOSSIBLE; j + 1]).collect() } else { Vec::new() }
    };
    jbeta.push(el_rows(cp9b.jvalid[m]));
    lbeta.push(el_rows(fill_l && cp9b.lvalid[m]));
    rbeta.push(el_rows(fill_r && cp9b.rvalid[m]));
    tbeta.push(Vec::new());

    // ROOT_S full-seq cell = 0.0 in the preset plane (C 8000-8005).
    match preset_mode {
        TRMODE_J => jbeta[0][jp_0][lp_0] = 0.0,
        TRMODE_L => lbeta[0][jp_0][lp_0] = 0.0,
        TRMODE_R => rbeta[0][jp_0][lp_0] = 0.0,
        TRMODE_T => tbeta[0][jp_0][lp_0] = 0.0,
        _ => {}
    }
    // legal truncated-begin entry states get the penalty (C 8012-8033).
    for v in 0..m {
        if l >= jmin[v] && l <= jmax[v] {
            let jp_v = (l - jmin[v]) as usize;
            if l >= hdmin[v][jp_v] && l <= hdmax[v][jp_v] {
                let lpp = (l - hdmin[v][jp_v]) as usize;
                let trpenalty = pty[v];
                if not_impossible(trpenalty) {
                    let do_j_v = cp9b.jvalid[v];
                    let do_l_v = cp9b.lvalid[v] && fill_l;
                    let do_r_v = cp9b.rvalid[v] && fill_r;
                    let do_t_v = cp9b.tvalid[v] && fill_t;
                    match preset_mode {
                        TRMODE_J => if do_j_v { jbeta[v][jp_v][lpp] = trpenalty; },
                        TRMODE_L => if do_l_v { lbeta[v][jp_v][lpp] = trpenalty; },
                        TRMODE_R => if do_r_v { rbeta[v][jp_v][lpp] = trpenalty; },
                        TRMODE_T => if do_t_v && cm.sttype[v] as i32 == B_ST { tbeta[v][jp_v][lpp] = trpenalty; },
                        _ => {}
                    }
                }
            }
        }
    }

    // ---- main loop v = 1 .. M-1 (C 8046) ----
    for v in 1..m {
        if state_is_detached(cm, v) { continue; }
        let sd = tr_state_delta(cm.sttype[v] as i32);
        let sdr = tr_state_right_delta(cm.sttype[v] as i32);
        let do_j_v = cp9b.jvalid[v];
        let do_l_v = cp9b.lvalid[v] && fill_l;
        let do_r_v = cp9b.rvalid[v] && fill_r;
        let do_t_v = cp9b.tvalid[v] && fill_t;
        if !(do_j_v || do_l_v || do_r_v || do_t_v) { continue; }
        let stid = cm.stid[v] as i32;

        if stid == BEGL_S {
            let y = cm.plast[v] as usize; // parent BIF_B
            let z = cm.cnum[y] as usize; // right child S
            let do_j_y = cp9b.jvalid[y];
            let do_l_y = cp9b.lvalid[y] && fill_l;
            let do_r_y = cp9b.rvalid[y] && fill_r;
            let do_t_y = cp9b.tvalid[y] && fill_t;
            let do_j_z = cp9b.jvalid[z];
            let do_l_z = cp9b.lvalid[z] && fill_l;
            for j in (jmin[v]..=jmax[v]).rev() {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = (j - jmin[y]) as i32;
                let jp_z = (j - jmin[z]) as i32;
                for d in (hdmin[v][jp_v]..=hdmax[v][jp_v]).rev() {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let kmin = jmin[y].max(jmin[z]) - j;
                    let kmax = jmax[y].min(jmax[z]) - j;
                    let mut k = kmin;
                    while k <= kmax {
                        let row_y = (jp_y + k) as usize;
                        let row_z = (jp_z + k) as usize;
                        if k >= hdmin[y][row_y] - d && k <= hdmax[y][row_y] - d
                            && k >= hdmin[z][row_z] && k <= hdmax[z][row_z]
                        {
                            let kp_z = (k - hdmin[z][row_z]) as usize;
                            // dp_y is SIGNED: d can be < hdmin[y][row_y] (guard is on col_y=dp_y+k>=0).
                            let dp_y = d - hdmin[y][row_y];
                            let col_y = (dp_y + k) as usize;
                            if do_j_v && do_j_y && do_j_z {
                                jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], jbeta[y][row_y][col_y] + ins.j[z][row_z][kp_z]);
                            }
                            if do_j_v && do_l_y && do_l_z {
                                jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], lbeta[y][row_y][col_y] + ins.l[z][row_z][kp_z]);
                            }
                            if do_r_v && do_r_y && do_j_z {
                                rbeta[v][jp_v][dp_v] = flogsum(rbeta[v][jp_v][dp_v], rbeta[y][row_y][col_y] + ins.j[z][row_z][kp_z]);
                            }
                            if d == j && (j + k) == l && do_r_v && do_t_y && do_l_z {
                                rbeta[v][jp_v][dp_v] = flogsum(rbeta[v][jp_v][dp_v], tbeta[y][row_y][col_y] + ins.l[z][row_z][kp_z]);
                            }
                        }
                        k += 1;
                    }
                }
            }
            // k==0 special: entire seq on left (C 8135-8151)
            if do_l_y && (do_j_v || do_l_v) {
                let jn = jmin[v].max(jmin[y]);
                let jx = jmax[v].min(jmax[y]);
                for j in (jn..=jx).rev() {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_y = (j - jmin[y]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    for d in (dn..=dx).rev() {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        if do_j_v { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], lbeta[y][jp_y][dp_y]); }
                        if do_l_v { lbeta[v][jp_v][dp_v] = flogsum(lbeta[v][jp_v][dp_v], lbeta[y][jp_y][dp_y]); }
                    }
                }
            }
        } else if stid == BEGR_S {
            let y = cm.plast[v] as usize; // parent BIF_B
            let z = cm.cfirst[y] as usize; // left child S
            let do_j_y = cp9b.jvalid[y];
            let do_r_y = cp9b.rvalid[y] && fill_r;
            let do_t_y = cp9b.tvalid[y] && fill_t;
            let do_j_z = cp9b.jvalid[z];
            let do_r_z = cp9b.rvalid[z] && fill_r;
            let jn = jmin[v].max(jmin[y]);
            let jx = jmax[v].min(jmax[y]);
            for j in (jn..=jx).rev() {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = (j - jmin[y]) as usize;
                let jp_z = (j - jmin[z]) as i32;
                let dn = hdmin[v][jp_v].max(j - jmax[z]);
                let dx = hdmax[v][jp_v].min(jp_z);
                for d in (dn..=dx).rev() {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let row_z = (jp_z - d) as usize;
                    let kmin = (hdmin[y][jp_y] - d).max(hdmin[z][row_z]);
                    let kmax = (hdmax[y][jp_y] - d).min(hdmax[z][row_z]);
                    let i = j - d + 1;
                    let mut k = kmin;
                    while k <= kmax {
                        let kp_z = (k - hdmin[z][row_z]) as usize;
                        // dp_y is SIGNED: d can be < hdmin[y][jp_y] (guard is on col_y=dp_y+k>=0).
                        let dp_y = d - hdmin[y][jp_y];
                        let col_y = (dp_y + k) as usize;
                        if do_j_v && do_j_y && do_j_z {
                            jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], jbeta[y][jp_y][col_y] + ins.j[z][row_z][kp_z]);
                        }
                        if do_j_v && do_r_y && do_r_z {
                            jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], rbeta[y][jp_y][col_y] + ins.r[z][row_z][kp_z]);
                        }
                        if do_l_v && (cp9b.lvalid[y] && fill_l) && do_j_z {
                            lbeta[v][jp_v][dp_v] = flogsum(lbeta[v][jp_v][dp_v], lbeta[y][jp_y][col_y] + ins.j[z][row_z][kp_z]);
                        }
                        if k == (i as i32 - 1) && j == l && do_l_v && do_t_y && do_r_z {
                            lbeta[v][jp_v][dp_v] = flogsum(lbeta[v][jp_v][dp_v], tbeta[y][jp_y][col_y] + ins.r[z][row_z][kp_z]);
                        }
                        k += 1;
                    }
                }
                // k==0 special: entire seq on right (C 8241-8250)
                if do_r_y && (do_j_v || do_r_v) {
                    let dn2 = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx2 = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    for d in (dn2..=dx2).rev() {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        let dp_y = (d - hdmin[y][jp_y]) as usize;
                        if do_j_v { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], rbeta[y][jp_y][dp_y]); }
                        if do_r_v { rbeta[v][jp_v][dp_v] = flogsum(rbeta[v][jp_v][dp_v], rbeta[y][jp_y][dp_y]); }
                    }
                }
            }
        } else {
            // general states: iterate parents y = plast[v] .. plast[v]-pnum[v]+1 (C 8256)
            let plast_v = cm.plast[v];
            let pnum_v = cm.pnum[v];
            for j in (jmin[v]..=jmax[v]).rev() {
                let jp_v = (j - jmin[v]) as usize;
                for d in (hdmin[v][jp_v]..=hdmax[v][jp_v]).rev() {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let i = j - d + 1;
                    let mut yy = plast_v;
                    while yy > plast_v - pnum_v {
                        let y = yy as usize;
                        if y != 0 {
                            let voffset = (v as i32 - cm.cfirst[y]) as usize;
                            let yst = cm.sttype[y] as i32;
                            let sdy = tr_state_delta(yst);
                            let sdly = tr_state_left_delta(yst);
                            let sdry = tr_state_right_delta(yst);
                            let tscy = cm.tsc[y][voffset];
                            let esc_y = &cm.oesc[y];
                            let do_j_y = cp9b.jvalid[y];
                            let do_l_y = cp9b.lvalid[y] && fill_l;
                            let do_r_y = cp9b.rvalid[y] && fill_r;
                            if do_j_y || do_l_y || do_r_y {
                                let jp_y = j - jmin[y];
                                match yst {
                                    x if x == MP_ST => {
                                        if j != l && d != j && do_j_v && do_j_y
                                            && (j + sdry >= jmin[y] && j + sdry <= jmax[y])
                                        {
                                            let jr = (jp_y + sdry) as usize;
                                            if d + sdy >= hdmin[y][jr] && d + sdy <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                let escore = esc_y[dsq[(i - 1) as usize] as usize * kp + dsq[(j + 1) as usize] as usize];
                                                jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], jbeta[y][jr][(dp_y + sdy) as usize] + tscy + escore);
                                            }
                                        }
                                        if j == l && d != j && do_l_y
                                            && (j >= jmin[y] && j <= jmax[y])
                                        {
                                            let jr = jp_y as usize;
                                            if d + sdly >= hdmin[y][jr] && d + sdly <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                let escore = lm[y][dsq[(i - 1) as usize] as usize];
                                                if do_j_v { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], lbeta[y][jr][(dp_y + sdly) as usize] + tscy + escore); }
                                                if do_l_v { lbeta[v][jp_v][dp_v] = flogsum(lbeta[v][jp_v][dp_v], lbeta[y][jr][(dp_y + sdly) as usize] + tscy + escore); }
                                            }
                                        }
                                        if i == 1 && j != l && do_r_y
                                            && (j + sdry >= jmin[y] && j + sdry <= jmax[y])
                                        {
                                            let jr = (jp_y + sdry) as usize;
                                            if d + sdry >= hdmin[y][jr] && d + sdry <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                let escore = rm[y][dsq[(j + 1) as usize] as usize];
                                                if do_j_v { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], rbeta[y][jr][(dp_y + sdry) as usize] + tscy + escore); }
                                                if do_r_v { rbeta[v][jp_v][dp_v] = flogsum(rbeta[v][jp_v][dp_v], rbeta[y][jr][(dp_y + sdry) as usize] + tscy + escore); }
                                            }
                                        }
                                    }
                                    x if x == ML_ST || x == IL_ST => {
                                        if d != j && (j >= jmin[y] && j <= jmax[y]) {
                                            let jr = jp_y as usize;
                                            if d + sdly >= hdmin[y][jr] && d + sdly <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                let escore = esc_y[dsq[(i - 1) as usize] as usize];
                                                if do_j_v && do_j_y { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], jbeta[y][jr][(dp_y + sdy) as usize] + tscy + escore); }
                                                if do_l_v && do_l_y { lbeta[v][jp_v][dp_v] = flogsum(lbeta[v][jp_v][dp_v], lbeta[y][jr][(dp_y + sdy) as usize] + tscy + escore); }
                                            }
                                        }
                                        if i == 1 && v != y && do_r_y
                                            && (j >= jmin[y] && j <= jmax[y])
                                        {
                                            let jr = jp_y as usize;
                                            if d >= hdmin[y][jr] && d <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                if do_j_v { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], rbeta[y][jr][dp_y as usize] + tscy); }
                                                if do_r_v { rbeta[v][jp_v][dp_v] = flogsum(rbeta[v][jp_v][dp_v], rbeta[y][jr][dp_y as usize] + tscy); }
                                            }
                                        }
                                    }
                                    x if x == MR_ST || x == IR_ST => {
                                        if j != l && (j + sdry >= jmin[y] && j + sdry <= jmax[y]) {
                                            let jr = (jp_y + sdry) as usize;
                                            if d + sdy >= hdmin[y][jr] && d + sdy <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                let escore = esc_y[dsq[(j + 1) as usize] as usize];
                                                if do_j_v && do_j_y { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], jbeta[y][jr][(dp_y + sdy) as usize] + tscy + escore); }
                                                if do_r_v && do_r_y { rbeta[v][jp_v][dp_v] = flogsum(rbeta[v][jp_v][dp_v], rbeta[y][jr][(dp_y + sdy) as usize] + tscy + escore); }
                                            }
                                        }
                                        if j == l && v != y && do_l_y
                                            && (j >= jmin[y] && j <= jmax[y])
                                        {
                                            let jr = jp_y as usize;
                                            if d >= hdmin[y][jr] && d <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                if do_j_v { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], lbeta[y][jr][dp_y as usize] + tscy); }
                                                if do_l_v { lbeta[v][jp_v][dp_v] = flogsum(lbeta[v][jp_v][dp_v], lbeta[y][jr][dp_y as usize] + tscy); }
                                            }
                                        }
                                    }
                                    _ => {
                                        // S, E, D
                                        if j >= jmin[y] && j <= jmax[y] {
                                            let jr = jp_y as usize;
                                            if d >= hdmin[y][jr] && d <= hdmax[y][jr] {
                                                let dp_y = d - hdmin[y][jr];
                                                if do_j_v && do_j_y { jbeta[v][jp_v][dp_v] = flogsum(jbeta[v][jp_v][dp_v], jbeta[y][jr][dp_y as usize] + tscy); }
                                                if do_l_v && do_l_y { lbeta[v][jp_v][dp_v] = flogsum(lbeta[v][jp_v][dp_v], lbeta[y][jr][dp_y as usize] + tscy); }
                                                if do_r_v && do_r_y { rbeta[v][jp_v][dp_v] = flogsum(rbeta[v][jp_v][dp_v], rbeta[y][jr][dp_y as usize] + tscy); }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        yy -= 1;
                    }
                    if do_j_v && jbeta[v][jp_v][dp_v] < IMPOSSIBLE { jbeta[v][jp_v][dp_v] = IMPOSSIBLE; }
                    if do_l_v && lbeta[v][jp_v][dp_v] < IMPOSSIBLE { lbeta[v][jp_v][dp_v] = IMPOSSIBLE; }
                    if do_r_v && rbeta[v][jp_v][dp_v] < IMPOSSIBLE { rbeta[v][jp_v][dp_v] = IMPOSSIBLE; }
                }
            }
        }

        // ---- local end transitions v -> EL (deck M) (C 8398-8558) ----
        if local_end && not_impossible(cm.endsc[v]) {
            let vst = cm.sttype[v] as i32;
            let sdl = tr_state_left_delta(vst);
            let endsc = cm.endsc[v];
            let esc_v = &cm.oesc[v];
            // emitmode: EMITPAIR(MP), EMITLEFT(ML/IL), EMITRIGHT(MR/IR), EMITNONE(else)
            let em_pair = vst == MP_ST;
            let em_left = vst == ML_ST || vst == IL_ST;
            let em_right = vst == MR_ST || vst == IR_ST;
            // J mode. jp_v/dp_v are SIGNED: the loop starts below jmin[v] (jp_v<0)
            // and the EL dest [j][d] can have j<0 or d<0. In C those source cells are
            // IMPOSSIBLE (a state can't emit sd residues from a d<sd cell), so the
            // FLogsum is a no-op; we guard the dest write with j>=0 && d>=0.
            if do_j_v && cp9b.jvalid[m] {
                let jn = jmin[v] - sdr;
                let jx = jmax[v] - sdr;
                for j in jn..=jx {
                    let jp_v = j - jmin[v];
                    let row = (jp_v + sdr) as usize;
                    let dn = hdmin[v][row] - sd;
                    let dx = hdmax[v][row] - sd;
                    let mut i = j - dn + 1;
                    let mut dp_v: i32 = dn - hdmin[v][row];
                    for d in dn..=dx {
                        let base = jbeta[v][row][(dp_v + sd) as usize] + endsc;
                        let add = if em_pair {
                            base + esc_v[dsq[(i - 1) as usize] as usize * kp + dsq[(j + 1) as usize] as usize]
                        } else if em_left {
                            base + esc_v[dsq[(i - 1) as usize] as usize]
                        } else if em_right {
                            base + esc_v[dsq[(j + 1) as usize] as usize]
                        } else {
                            base
                        };
                        if j >= 0 && d >= 0 {
                            jbeta[m][j as usize][d as usize] = flogsum(jbeta[m][j as usize][d as usize], add);
                        }
                        dp_v += 1;
                        i -= 1;
                    }
                }
            }
            // L mode
            if do_l_v && cp9b.lvalid[m] {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let dn = hdmin[v][jp_v] - sdl;
                    let dx = hdmax[v][jp_v] - sdl;
                    let mut i = j - dn + 1;
                    let mut dp_v: i32 = dn - hdmin[v][jp_v];
                    for d in dn..=dx {
                        if em_pair {
                            if j == l && j >= 0 && d >= 0 {
                                let base = lbeta[v][jp_v][(dp_v + sdl) as usize] + endsc;
                                let escore = lm[v][dsq[(i - 1) as usize] as usize];
                                lbeta[m][j as usize][d as usize] = flogsum(lbeta[m][j as usize][d as usize], base + escore);
                            }
                        } else if em_left {
                            if j >= 0 && d >= 0 {
                                let base = lbeta[v][jp_v][(dp_v + sdl) as usize] + endsc;
                                let escore = esc_v[dsq[(i - 1) as usize] as usize];
                                lbeta[m][j as usize][d as usize] = flogsum(lbeta[m][j as usize][d as usize], base + escore);
                            }
                        } else if em_right {
                            if j == l && j >= 0 && d >= 0 {
                                // C uses dp_v (not dp_v+sdl) here for EMITRIGHT L-mode (sdl==0)
                                let b2 = lbeta[v][jp_v][dp_v as usize] + endsc;
                                lbeta[m][j as usize][d as usize] = flogsum(lbeta[m][j as usize][d as usize], b2);
                            }
                        } else if j >= 0 && d >= 0 {
                            let base = lbeta[v][jp_v][(dp_v + sdl) as usize] + endsc;
                            lbeta[m][j as usize][d as usize] = flogsum(lbeta[m][j as usize][d as usize], base);
                        }
                        dp_v += 1;
                        i -= 1;
                    }
                }
            }
            // R mode
            if do_r_v && cp9b.rvalid[m] {
                let jn = jmin[v] - sdr;
                let jx = jmax[v] - sdr;
                for j in jn..=jx {
                    let jp_v = j - jmin[v];
                    let row = (jp_v + sdr) as usize;
                    let dn = hdmin[v][row] - sdr;
                    let dx = hdmax[v][row] - sdr;
                    let mut i = j - dn + 1;
                    let mut dp_v: i32 = dn - hdmin[v][row];
                    for d in dn..=dx {
                        if em_pair {
                            if i == 1 && j >= 0 && d >= 0 {
                                let escore = rm[v][dsq[(j + 1) as usize] as usize];
                                rbeta[m][j as usize][d as usize] = flogsum(rbeta[m][j as usize][d as usize], rbeta[v][row][(dp_v + sdr) as usize] + endsc + escore);
                            }
                        } else if em_left {
                            if i == 1 && j >= 0 && d >= 0 {
                                rbeta[m][j as usize][d as usize] = flogsum(rbeta[m][j as usize][d as usize], rbeta[v][(jp_v) as usize][dp_v as usize] + endsc);
                            }
                        } else if em_right {
                            if j >= 0 && d >= 0 {
                                let escore = esc_v[dsq[(j + 1) as usize] as usize];
                                rbeta[m][j as usize][d as usize] = flogsum(rbeta[m][j as usize][d as usize], rbeta[v][row][(dp_v + sdr) as usize] + endsc + escore);
                            }
                        } else if j >= 0 && d >= 0 {
                            rbeta[m][j as usize][d as usize] = flogsum(rbeta[m][j as usize][d as usize], rbeta[v][row][(dp_v + sdr) as usize] + endsc);
                        }
                        dp_v += 1;
                        i -= 1;
                    }
                }
            }
        }
    }

    // EL->EL left-emitting transitions (C 8532-8558).
    if local_end {
        let elself = cm.el_selfsc;
        if cp9b.jvalid[m] {
            for j in (1..=lu).rev() {
                for d in (0..=(j - 1)).rev() {
                    jbeta[m][j][d] = flogsum(jbeta[m][j][d], jbeta[m][j][d + 1] + elself);
                }
            }
        }
        if fill_l && cp9b.lvalid[m] {
            for j in (1..=lu).rev() {
                for d in (0..=(j - 1)).rev() {
                    lbeta[m][j][d] = flogsum(lbeta[m][j][d], lbeta[m][j][d + 1] + elself);
                }
            }
        }
        if fill_r && cp9b.rvalid[m] {
            for j in (1..=lu).rev() {
                for d in (0..=(j - 1)).rev() {
                    rbeta[m][j][d] = flogsum(rbeta[m][j][d], rbeta[m][j][d + 1] + elself);
                }
            }
        }
    }

    TrHbMx { j: jbeta, l: lbeta, r: rbeta, t: tbeta }
}

/// C `cm_TrPosteriorHB` (cm_dpalign_trunc.c:8811): post = ins + out - optsc per
/// plane/cell over the banded decks + the non-banded EL deck (v==cm.M). optsc is the
/// Inside score in `preset_mode` at ROOT_S [L][L].
pub fn tr_posterior_hb(
    cm: &CM,
    cp9b: &crate::cp9::CP9Bands,
    lp: i32,
    preset_mode: i8,
    ins: &TrHbMx,
    out: &TrHbMx,
) -> TrHbMx {
    let m = cm.m as usize;
    let l = lp;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let (fill_l, fill_r, fill_t) = tr_fill_from_mode(preset_mode);
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;
    let sc = match preset_mode {
        TRMODE_J => ins.j[0][jp_0][lp_0],
        TRMODE_L => ins.l[0][jp_0][lp_0],
        TRMODE_R => ins.r[0][jp_0][lp_0],
        _ => ins.t[0][jp_0][lp_0],
    };

    // allocate post decks (same shape as ins/out).
    let njv = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut jp: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut lpm: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut rp: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut tp: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    for v in 0..m {
        let nj = njv(v);
        let is_b = cm.sttype[v] as i32 == B_ST;
        let mut jd = Vec::with_capacity(nj);
        let mut ld = Vec::with_capacity(nj);
        let mut rd = Vec::with_capacity(nj);
        let mut td = Vec::with_capacity(nj);
        for j in 0..nj {
            let w = (hdmax[v][j] - hdmin[v][j] + 1).max(0) as usize;
            jd.push(vec![IMPOSSIBLE; w]);
            ld.push(vec![IMPOSSIBLE; w]);
            rd.push(vec![IMPOSSIBLE; w]);
            if is_b || v == 0 { td.push(vec![IMPOSSIBLE; w]); }
        }
        jp.push(jd); lpm.push(ld); rp.push(rd); tp.push(td);
    }
    let lu = l as usize;
    let el_rows = |valid: bool| -> Vec<Vec<f32>> {
        if valid { (0..=lu).map(|j| vec![IMPOSSIBLE; j + 1]).collect() } else { Vec::new() }
    };
    jp.push(el_rows(cp9b.jvalid[m]));
    lpm.push(el_rows(fill_l && cp9b.lvalid[m]));
    rp.push(el_rows(fill_r && cp9b.rvalid[m]));
    tp.push(Vec::new());

    // EL deck (C 8858-8875).
    if local_end {
        if cp9b.jvalid[m] {
            for j in 0..=lu { for d in 0..=j { jp[m][j][d] = ins.j[m][j][d] + out.j[m][j][d] - sc; } }
        }
        if fill_l && cp9b.lvalid[m] {
            for j in 0..=lu { for d in 0..=j { lpm[m][j][d] = ins.l[m][j][d] + out.l[m][j][d] - sc; } }
        }
        if fill_r && cp9b.rvalid[m] {
            for j in 0..=lu { for d in 0..=j { rp[m][j][d] = ins.r[m][j][d] + out.r[m][j][d] - sc; } }
        }
    }
    // banded decks J, then L, R, T (C 8878-8935).
    for v in (0..m).rev() {
        if cp9b.jvalid[v] {
            for jpv in 0..njv(v) {
                let w = (hdmax[v][jpv] - hdmin[v][jpv] + 1).max(0) as usize;
                for dp in 0..w { jp[v][jpv][dp] = ins.j[v][jpv][dp] + out.j[v][jpv][dp] - sc; }
            }
        }
    }
    if fill_l {
        for v in (0..m).rev() {
            if cp9b.lvalid[v] {
                for jpv in 0..njv(v) {
                    let w = (hdmax[v][jpv] - hdmin[v][jpv] + 1).max(0) as usize;
                    for dp in 0..w { lpm[v][jpv][dp] = ins.l[v][jpv][dp] + out.l[v][jpv][dp] - sc; }
                }
            }
        }
    }
    if fill_r {
        for v in (0..m).rev() {
            if cp9b.rvalid[v] {
                for jpv in 0..njv(v) {
                    let w = (hdmax[v][jpv] - hdmin[v][jpv] + 1).max(0) as usize;
                    for dp in 0..w { rp[v][jpv][dp] = ins.r[v][jpv][dp] + out.r[v][jpv][dp] - sc; }
                }
            }
        }
    }
    if fill_t {
        for v in (0..m).rev() {
            if cp9b.tvalid[v] && (v == 0 || cm.sttype[v] as i32 == B_ST) {
                for jpv in 0..njv(v) {
                    let w = (hdmax[v][jpv] - hdmin[v][jpv] + 1).max(0) as usize;
                    for dp in 0..w { tp[v][jpv][dp] = ins.t[v][jpv][dp] + out.t[v][jpv][dp] - sc; }
                }
            }
        }
    }
    TrHbMx { j: jp, l: lpm, r: rp, t: tp }
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
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
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

// ============================================================================
// Truncated non-banded posterior + OptAcc stack (cm_dpalign_trunc.c).
// Flat per-plane DP cube: plane[v*ncell + j*stride + d], v in 0..=M (deck M = EL),
// stride = L+1, ncell = stride*stride. All four marginal planes (J/L/R/T) are
// allocated full-size; the T plane is only meaningfully written for B states and
// ROOT (v==0), matching C's access pattern.
// ============================================================================

/// C `CM_TR_MX` (float 4-plane DP cube), flat representation.
pub struct TrMx {
    pub j: Vec<f32>,
    pub l: Vec<f32>,
    pub r: Vec<f32>,
    pub t: Vec<f32>,
    pub stride: usize,
    pub ncell: usize,
    pub m: usize, // cm->M (EL deck index)
}
impl TrMx {
    fn new(m: usize, ll: usize, init: f32) -> Self {
        let stride = ll + 1;
        let ncell = stride * stride;
        let sz = (m + 1) * ncell;
        TrMx { j: vec![init; sz], l: vec![init; sz], r: vec![init; sz], t: vec![init; sz], stride, ncell, m }
    }
    #[inline]
    fn base(&self, v: usize) -> usize {
        v * self.ncell
    }
}

/// C `cm_TrFillFromMode` (cm_dpalign_trunc.c:9858): which of the L/R/T planes to fill.
#[inline]
fn tr_fill_from_mode(mode: i8) -> (bool, bool, bool) {
    match mode {
        TRMODE_J => (false, false, false),
        TRMODE_L => (true, false, false),
        TRMODE_R => (false, true, false),
        _ => (true, true, true), // TRMODE_T or TRMODE_UNKNOWN
    }
}

/// C `cm_TrInsideAlign` (cm_dpalign_trunc.c:2970): full truncated Inside DP over the
/// J/L/R/T marginal planes; retains the matrix and returns the winning mode. Uses
/// FLogsum (not max). Supports local ends (EL deck) + local begins (use_local penalty).
/// `preset_mode` is TRMODE_UNKNOWN for cmalign (fill all planes, pick max-scoring mode).
/// Emission scheme mirrors the byte-verified `tr_cyk_align`/`tr_inside_score`
/// (cm.esc + sing_sc/pair_sc, marginal lm/rm + marg_sc).
#[allow(clippy::too_many_arguments)]
pub fn tr_inside_align(
    cm: &CM,
    trp: &TrPenalties,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    lp: i32,
    pass_idx: i32,
    preset_mode: i8,
    use_local: bool,
) -> (TrMx, i8, f32) {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let ncell = stride * stride;
    let pty = trp.pty_slice(pass_idx, use_local);
    let (fill_l, fill_r, fill_t) = tr_fill_from_mode(preset_mode);
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let mut mx = TrMx::new(m, w, IMPOSSIBLE);

    // C 3013-3015: el_scA[d] = el_selfsc * d.
    let mut el_sca = vec![0.0f32; w + 1];
    for d in 0..=w {
        el_sca[d] = cm.el_selfsc * d as f32;
    }
    // C 3023-3044: if local ends on, EL deck (v==M) = el_scA[d].
    if local_end {
        let mb = mx.base(m);
        for j in 0..=w {
            let jrow = j * stride;
            for d in 0..=j {
                mx.j[mb + jrow + d] = el_sca[d];
                if fill_l {
                    mx.l[mb + jrow + d] = el_sca[d];
                }
                if fill_r {
                    mx.r[mb + jrow + d] = el_sca[d];
                }
            }
        }
    }

    // C 3047: main recursion v = M-1 downto 1.
    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = tr_state_delta(stt) as usize;
        let sdl = tr_state_left_delta(stt) as usize;
        let sdr = tr_state_right_delta(stt) as usize;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];
        let endsc_v = cm.endsc[v];
        let vb = mx.base(v);
        let mb = mx.base(m);

        // C 3056-3077: re-init J/L/R decks if we can do a local end from v.
        if not_impossible(endsc_v) {
            for j in 0..=w {
                let jrow = j * stride;
                for d in sd..=j {
                    mx.j[vb + jrow + d] = mx.j[mb + jrow + (d - sd)] + endsc_v;
                }
                if fill_l {
                    for d in sdl..=j {
                        mx.l[vb + jrow + d] = mx.l[mb + jrow + (d - sdl)] + endsc_v;
                    }
                }
                if fill_r {
                    for d in sdr..=j {
                        mx.r[vb + jrow + d] = mx.r[mb + jrow + (d - sdr)] + endsc_v;
                    }
                }
            }
        }

        if stt == E_ST {
            // C 3080-3086
            for j in 0..=w {
                mx.j[vb + j * stride] = 0.0;
                if fill_l {
                    mx.l[vb + j * stride] = 0.0;
                }
                if fill_r {
                    mx.r[vb + j * stride] = 0.0;
                }
            }
        } else if stt == IL_ST || stt == ML_ST {
            // C 3088-3133
            if !state_is_detached(cm, v) {
                let ryoffset0 = if stt == IL_ST { 1 } else { 0 };
                for j in sdr..=w {
                    let j_sdr = j - sdr;
                    for d in sd..=j {
                        let d_sd = d - sd;
                        let i = j - d + 1;
                        let idx = j * stride + d;
                        // J and L (use j_sdr, d_sd)
                        let mut jv = mx.j[vb + idx];
                        let mut lv = mx.l[vb + idx];
                        let src = j_sdr * stride + d_sd;
                        for yo in 0..cnum {
                            let y = cfirst + yo;
                            let yb = mx.base(y);
                            jv = flogsum(jv, mx.j[yb + src] + tsc_v[yo]);
                            if fill_l {
                                lv = flogsum(lv, mx.l[yb + src] + tsc_v[yo]);
                            }
                        }
                        let e = sing_sc(esc_v, dsq[i]);
                        jv = (jv + e).max(IMPOSSIBLE);
                        mx.j[vb + idx] = jv;
                        if fill_l {
                            lv = if d >= 2 { lv + e } else { e };
                            mx.l[vb + idx] = lv.max(IMPOSSIBLE);
                        }
                        // R (use j_sdr, d) — separate loop
                        if fill_r {
                            let mut rv = mx.r[vb + idx];
                            let rsrc = j_sdr * stride + d;
                            for yo in ryoffset0..cnum {
                                let y = cfirst + yo;
                                let yb = mx.base(y);
                                rv = flogsum(rv, mx.j[yb + rsrc] + tsc_v[yo]);
                                rv = flogsum(rv, mx.r[yb + rsrc] + tsc_v[yo]);
                            }
                            mx.r[vb + idx] = rv.max(IMPOSSIBLE);
                        }
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            // C 3135-3181
            if !state_is_detached(cm, v) {
                let lyoffset0 = if stt == IR_ST { 1 } else { 0 };
                for j in sdr..=w {
                    let j_sdr = j - sdr;
                    for d in sd..=j {
                        let d_sd = d - sd;
                        let idx = j * stride + d;
                        let src = j_sdr * stride + d_sd;
                        let mut jv = mx.j[vb + idx];
                        let mut rv = mx.r[vb + idx];
                        for yo in 0..cnum {
                            let y = cfirst + yo;
                            let yb = mx.base(y);
                            jv = flogsum(jv, mx.j[yb + src] + tsc_v[yo]);
                            if fill_r {
                                rv = flogsum(rv, mx.r[yb + src] + tsc_v[yo]);
                            }
                        }
                        let e = sing_sc(esc_v, dsq[j]);
                        jv = (jv + e).max(IMPOSSIBLE);
                        mx.j[vb + idx] = jv;
                        if fill_r {
                            rv = if d >= 2 { rv + e } else { e };
                            mx.r[vb + idx] = rv.max(IMPOSSIBLE);
                        }
                        // L (use j, d) — separate loop
                        if fill_l {
                            let mut lv = mx.l[vb + idx];
                            let lsrc = j * stride + d;
                            for yo in lyoffset0..cnum {
                                let y = cfirst + yo;
                                let yb = mx.base(y);
                                lv = flogsum(lv, mx.j[yb + lsrc] + tsc_v[yo]);
                                lv = flogsum(lv, mx.l[yb + lsrc] + tsc_v[yo]);
                            }
                            mx.l[vb + idx] = lv.max(IMPOSSIBLE);
                        }
                    }
                }
            }
        } else if stt == MP_ST {
            // C 3182-3239
            let lmv = &lm[v];
            let rmv = &rm[v];
            for yo in 0..cnum {
                let y = cfirst + yo;
                let yb = mx.base(y);
                let tsc = tsc_v[yo];
                for j in 1..=w {
                    let jrow = j * stride;
                    let jm1 = (j - 1) * stride;
                    // J: d >= 2, src (j-1, d-2)
                    for d in 2..=j {
                        let idx = jrow + d;
                        let val = flogsum(mx.j[vb + idx], mx.j[yb + jm1 + (d - 2)] + tsc);
                        mx.j[vb + idx] = val;
                    }
                    if fill_l {
                        for d in 1..=j {
                            let idx = jrow + d;
                            let src = jrow + (d - 1);
                            let mut lv = mx.l[vb + idx];
                            lv = flogsum(lv, mx.j[yb + src] + tsc);
                            lv = flogsum(lv, mx.l[yb + src] + tsc);
                            mx.l[vb + idx] = lv;
                        }
                    }
                    if fill_r {
                        for d in 1..=j {
                            let idx = jrow + d;
                            let src = jm1 + (d - 1);
                            let mut rv = mx.r[vb + idx];
                            rv = flogsum(rv, mx.j[yb + src] + tsc);
                            rv = flogsum(rv, mx.r[yb + src] + tsc);
                            mx.r[vb + idx] = rv;
                        }
                    }
                }
            }
            // emission (C 3218-3231)
            for j in 0..=w {
                let jrow = j * stride;
                mx.j[vb + jrow + 1] = IMPOSSIBLE;
                if j >= 1 {
                    if fill_l {
                        mx.l[vb + jrow + 1] = marg_sc(lmv, dsq[j]);
                    }
                    if fill_r {
                        mx.r[vb + jrow + 1] = marg_sc(rmv, dsq[j]);
                    }
                }
                let mut i = if j >= 1 { j - 1 } else { 0 };
                for d in 2..=j {
                    let idx = jrow + d;
                    mx.j[vb + idx] += pair_sc(esc_v, dsq[i], dsq[j]);
                    if fill_l {
                        mx.l[vb + idx] += marg_sc(lmv, dsq[i]);
                    }
                    if fill_r {
                        mx.r[vb + idx] += marg_sc(rmv, dsq[j]);
                    }
                    i = i.wrapping_sub(1);
                }
            }
            // ensure >= IMPOSSIBLE (C 3232-3239)
            for j in 0..=w {
                let jrow = j * stride;
                for d in 1..=j {
                    let idx = jrow + d;
                    mx.j[vb + idx] = mx.j[vb + idx].max(IMPOSSIBLE);
                    if fill_l {
                        mx.l[vb + idx] = mx.l[vb + idx].max(IMPOSSIBLE);
                    }
                    if fill_r {
                        mx.r[vb + idx] = mx.r[vb + idx].max(IMPOSSIBLE);
                    }
                }
            }
        } else if stt != B_ST {
            // D, S states (C 3241-3267)
            for yo in 0..cnum {
                let y = cfirst + yo;
                let yb = mx.base(y);
                let tsc = tsc_v[yo];
                for j in 0..=w {
                    let jrow = j * stride;
                    for d in 0..=j {
                        let idx = jrow + d;
                        mx.j[vb + idx] = flogsum(mx.j[vb + idx], mx.j[yb + idx] + tsc);
                        if fill_l {
                            mx.l[vb + idx] = flogsum(mx.l[vb + idx], mx.l[yb + idx] + tsc);
                        }
                        if fill_r {
                            mx.r[vb + idx] = flogsum(mx.r[vb + idx], mx.r[yb + idx] + tsc);
                        }
                    }
                    if fill_l {
                        mx.l[vb + jrow] = IMPOSSIBLE;
                    }
                    if fill_r {
                        mx.r[vb + jrow] = IMPOSSIBLE;
                    }
                }
            }
        } else {
            // B_st (C 3268-3298)
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            let yb = mx.base(y);
            let zb = mx.base(z);
            for j in 0..=w {
                let jrow = j * stride;
                for d in 0..=j {
                    let idx = jrow + d;
                    let mut jv = mx.j[vb + idx];
                    let mut lv = mx.l[vb + idx];
                    let mut rv = mx.r[vb + idx];
                    for k in 0..=d {
                        let ly = (j - k) * stride + (d - k);
                        let zk = jrow + k;
                        jv = flogsum(jv, mx.j[yb + ly] + mx.j[zb + zk]);
                        if fill_l {
                            lv = flogsum(lv, mx.j[yb + ly] + mx.l[zb + zk]);
                        }
                        if fill_r {
                            rv = flogsum(rv, mx.r[yb + ly] + mx.j[zb + zk]);
                        }
                    }
                    if fill_t {
                        let mut tv = mx.t[vb + idx];
                        for k in 1..d {
                            let ly = (j - k) * stride + (d - k);
                            let zk = jrow + k;
                            tv = flogsum(tv, mx.r[yb + ly] + mx.l[zb + zk]);
                        }
                        mx.t[vb + idx] = tv;
                    }
                    // special cases k==0 and k==d
                    if fill_l {
                        lv = flogsum(lv, mx.j[yb + idx]);
                        lv = flogsum(lv, mx.l[yb + idx]);
                        mx.l[vb + idx] = lv;
                    }
                    if fill_r {
                        rv = flogsum(rv, mx.j[zb + idx]);
                        rv = flogsum(rv, mx.r[zb + idx]);
                        mx.r[vb + idx] = rv;
                    }
                    mx.j[vb + idx] = jv;
                }
            }
        }

        // C 3300-3328: ROOT_S truncated-begin update at [0][L][L].
        let trpenalty = pty[v];
        if not_impossible(trpenalty) {
            let root = w * stride + w;
            let b0 = 0; // base(0) == 0
            mx.j[b0 + root] = flogsum(mx.j[b0 + root], mx.j[vb + root] + trpenalty);
            if fill_l {
                mx.l[b0 + root] = flogsum(mx.l[b0 + root], mx.l[vb + root] + trpenalty);
            }
            if fill_r {
                mx.r[b0 + root] = flogsum(mx.r[b0 + root], mx.r[vb + root] + trpenalty);
            }
            if fill_t && stt == B_ST {
                mx.t[b0 + root] = flogsum(mx.t[b0 + root], mx.t[vb + root] + trpenalty);
            }
        }
    }

    // C 3331-3363: determine mode of optimal alignment.
    let root = w * stride + w;
    let (mut sc, mut mode);
    match preset_mode {
        TRMODE_J => {
            sc = mx.j[root];
            mode = TRMODE_J;
        }
        TRMODE_L => {
            sc = mx.l[root];
            mode = TRMODE_L;
        }
        TRMODE_R => {
            sc = mx.r[root];
            mode = TRMODE_R;
        }
        TRMODE_T => {
            sc = mx.t[root];
            mode = TRMODE_T;
        }
        _ => {
            sc = mx.j[root];
            mode = TRMODE_J;
            if fill_l && mx.l[root] > sc {
                sc = mx.l[root];
                mode = TRMODE_L;
            }
            if fill_r && mx.r[root] > sc {
                sc = mx.r[root];
                mode = TRMODE_R;
            }
            if fill_t && mx.t[root] > sc {
                sc = mx.t[root];
                mode = TRMODE_T;
            }
        }
    }
    let _ = &mut sc;
    let _ = &mut mode;
    (mx, mode, sc)
}

/// C `cm_TrOutsideAlign` (cm_dpalign_trunc.c:7466): truncated Outside DP (FLogsum),
/// nonbanded. `preset_mode` is the determined optimal mode (J/L/R/T). Requires the
/// filled Inside matrix `ins`. do_check is not ported (CHECKINOUT off by default).
#[allow(clippy::too_many_arguments)]
pub fn tr_outside_align(
    cm: &CM,
    trp: &TrPenalties,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    lp: i32,
    preset_mode: i8,
    pass_idx: i32,
    use_local: bool,
    ins: &TrMx,
) -> TrMx {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let pty = trp.pty_slice(pass_idx, use_local);
    let (fill_l, fill_r, fill_t) = tr_fill_from_mode(preset_mode);
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let root = w * stride + w;

    let mut beta = TrMx::new(m, w, IMPOSSIBLE);

    // C 7519-7524: init ROOT_S full-seq cell to 0.0 in the preset mode's plane.
    match preset_mode {
        TRMODE_J => beta.j[root] = 0.0,
        TRMODE_L => beta.l[root] = 0.0,
        TRMODE_R => beta.r[root] = 0.0,
        TRMODE_T => beta.t[root] = 0.0,
        _ => {}
    }
    // C 7534-7544: legal truncated-begin entry states get the penalty.
    for v in 0..m {
        let trpenalty = pty[v];
        if not_impossible(trpenalty) {
            let vb = beta.base(v);
            match preset_mode {
                TRMODE_J => beta.j[vb + root] = trpenalty,
                TRMODE_L => beta.l[vb + root] = trpenalty,
                TRMODE_R => beta.r[vb + root] = trpenalty,
                TRMODE_T => {
                    if cm.sttype[v] as i32 == B_ST {
                        beta.t[vb + root] = trpenalty;
                    }
                }
                _ => {}
            }
        }
    }

    // C 7547: main loop v = 1 .. M-1.
    for v in 1..m {
        if !state_is_detached(cm, v) {
            let vb = beta.base(v);
            let stid = cm.stid[v] as i32;
            if stid == BEGL_S {
                // C 7552-7583
                let y = cm.plast[v] as usize; // parent bifurcation
                let z = cm.cnum[y] as usize; // the other (right) S state
                let yb = beta.base(y);
                let zb = ins.base(z);
                for j in 0..=w {
                    let jrow = j * stride;
                    for d in 0..=j {
                        let idx = jrow + d;
                        let mut jv = beta.j[vb + idx];
                        let mut rv = beta.r[vb + idx];
                        for k in 0..=(w - j) {
                            let ykd = (j + k) * stride + (d + k);
                            let zk = (j + k) * stride + k;
                            jv = flogsum(jv, beta.j[yb + ykd] + ins.j[zb + zk]);
                            if fill_l {
                                jv = flogsum(jv, beta.l[yb + ykd] + ins.l[zb + zk]);
                            }
                            if fill_r {
                                rv = flogsum(rv, beta.r[yb + ykd] + ins.j[zb + zk]);
                                if fill_t && fill_l && d == j && (j + k) == w {
                                    rv = flogsum(rv, beta.t[yb + ykd] + ins.l[zb + zk]);
                                }
                            }
                        }
                        beta.j[vb + idx] = jv;
                        if fill_r {
                            beta.r[vb + idx] = rv;
                        }
                        if fill_l {
                            // k == 0 special: entire sequence on left
                            let ljd = beta.l[yb + idx];
                            beta.j[vb + idx] = flogsum(beta.j[vb + idx], ljd);
                            beta.l[vb + idx] = flogsum(beta.l[vb + idx], ljd);
                        }
                    }
                }
            } else if stid == BEGR_S {
                // C 7584-7616
                let y = cm.plast[v] as usize; // parent bifurcation
                let z = cm.cfirst[y] as usize; // the other (left) S state
                let yb = beta.base(y);
                let zb = ins.base(z);
                for j in 0..=w {
                    let jrow = j * stride;
                    for d in 0..=j {
                        let idx = jrow + d;
                        let i = j - d + 1;
                        let mut jv = beta.j[vb + idx];
                        let mut lv = beta.l[vb + idx];
                        for k in 0..=(j - d) {
                            let ydk = jrow + (d + k);
                            let zk = (j - d) * stride + k;
                            jv = flogsum(jv, beta.j[yb + ydk] + ins.j[zb + zk]);
                            if fill_r {
                                jv = flogsum(jv, beta.r[yb + ydk] + ins.r[zb + zk]);
                            }
                            if fill_l {
                                lv = flogsum(lv, beta.l[yb + ydk] + ins.j[zb + zk]);
                                if fill_r && fill_t && k == (i - 1) && j == w {
                                    lv = flogsum(lv, beta.t[yb + ydk] + ins.r[zb + zk]);
                                }
                            }
                        }
                        beta.j[vb + idx] = jv;
                        if fill_l {
                            beta.l[vb + idx] = lv;
                        }
                        if fill_r {
                            // k == 0 special: entire sequence on right
                            let rjd = beta.r[yb + idx];
                            beta.j[vb + idx] = flogsum(beta.j[vb + idx], rjd);
                            beta.r[vb + idx] = flogsum(beta.r[vb + idx], rjd);
                        }
                    }
                }
            } else {
                // C 7617-7690: general states, iterate parents y.
                let plast_v = cm.plast[v];
                let pnum_v = cm.pnum[v];
                for j in (0..=w).rev() {
                    let jrow = j * stride;
                    let mut i = 1;
                    let mut d = j as i32;
                    while d >= 0 {
                        let du = d as usize;
                        let idx = jrow + du;
                        // iterate parents y = plast[v] down to plast[v]-pnum[v]+1
                        let mut yy = plast_v;
                        while yy > plast_v - pnum_v {
                            let y = yy as usize;
                            if y != 0 {
                                let yb = beta.base(y);
                                let voffset = (v as i32 - cm.cfirst[y]) as usize;
                                let yst = cm.sttype[y] as i32;
                                let sd = tr_state_delta(yst) as usize;
                                let sdl = tr_state_left_delta(yst) as usize;
                                let sdr = tr_state_right_delta(yst) as usize;
                                let tscy = cm.tsc[y][voffset];
                                let esc_y = &cm.esc[y];
                                match yst {
                                    x if x == MP_ST => {
                                        if (j as usize) != w && du != j as usize {
                                            let escore = pair_sc(esc_y, dsq[i - 1], dsq[j + 1]);
                                            let src = (j + sdr) * stride + (du + sd);
                                            beta.j[vb + idx] =
                                                flogsum(beta.j[vb + idx], beta.j[yb + src] + tscy + escore);
                                        }
                                        if fill_l && (j as usize) == w && du != j as usize {
                                            let escore = marg_sc(&lm[y], dsq[i - 1]);
                                            let src = jrow + (du + sdl);
                                            beta.j[vb + idx] =
                                                flogsum(beta.j[vb + idx], beta.l[yb + src] + tscy + escore);
                                            beta.l[vb + idx] =
                                                flogsum(beta.l[vb + idx], beta.l[yb + src] + tscy + escore);
                                        }
                                        if fill_r && i == 1 && (j as usize) != w {
                                            let escore = marg_sc(&rm[y], dsq[j + 1]);
                                            let src = (j + sdr) * stride + (du + sdr);
                                            beta.j[vb + idx] =
                                                flogsum(beta.j[vb + idx], beta.r[yb + src] + tscy + escore);
                                            beta.r[vb + idx] =
                                                flogsum(beta.r[vb + idx], beta.r[yb + src] + tscy + escore);
                                        }
                                    }
                                    x if x == ML_ST || x == IL_ST => {
                                        if du != j as usize {
                                            let escore = sing_sc(esc_y, dsq[i - 1]);
                                            let src = jrow + (du + sd);
                                            beta.j[vb + idx] =
                                                flogsum(beta.j[vb + idx], beta.j[yb + src] + tscy + escore);
                                            if fill_l {
                                                beta.l[vb + idx] =
                                                    flogsum(beta.l[vb + idx], beta.l[yb + src] + tscy + escore);
                                            }
                                        }
                                        if fill_r && i == 1 && v != y {
                                            beta.j[vb + idx] =
                                                flogsum(beta.j[vb + idx], beta.r[yb + idx] + tscy);
                                            beta.r[vb + idx] =
                                                flogsum(beta.r[vb + idx], beta.r[yb + idx] + tscy);
                                        }
                                    }
                                    x if x == MR_ST || x == IR_ST => {
                                        if (j as usize) != w {
                                            let escore = sing_sc(esc_y, dsq[j + 1]);
                                            let src = (j + sdr) * stride + (du + sd);
                                            beta.j[vb + idx] =
                                                flogsum(beta.j[vb + idx], beta.j[yb + src] + tscy + escore);
                                            if fill_r {
                                                beta.r[vb + idx] =
                                                    flogsum(beta.r[vb + idx], beta.r[yb + src] + tscy + escore);
                                            }
                                        }
                                        if fill_l && (j as usize) == w && v != y {
                                            beta.j[vb + idx] =
                                                flogsum(beta.j[vb + idx], beta.l[yb + idx] + tscy);
                                            beta.l[vb + idx] =
                                                flogsum(beta.l[vb + idx], beta.l[yb + idx] + tscy);
                                        }
                                    }
                                    _ => {
                                        // S, E, D
                                        beta.j[vb + idx] =
                                            flogsum(beta.j[vb + idx], beta.j[yb + idx] + tscy);
                                        if fill_l {
                                            beta.l[vb + idx] =
                                                flogsum(beta.l[vb + idx], beta.l[yb + idx] + tscy);
                                        }
                                        if fill_r {
                                            beta.r[vb + idx] =
                                                flogsum(beta.r[vb + idx], beta.r[yb + idx] + tscy);
                                        }
                                    }
                                }
                            }
                            yy -= 1;
                        }
                        if beta.j[vb + idx] < IMPOSSIBLE {
                            beta.j[vb + idx] = IMPOSSIBLE;
                        }
                        d -= 1;
                        i += 1;
                    }
                }
            }
        }

        // C 7695-7750: local end transitions v -> EL (deck M).
        if local_end && not_impossible(cm.endsc[v]) {
            let vb = beta.base(v);
            let mb = beta.base(m);
            let vst = cm.sttype[v] as i32;
            let sd = tr_state_delta(vst) as usize;
            let sdl = tr_state_left_delta(vst) as usize;
            let sdr = tr_state_right_delta(vst) as usize;
            let endsc = cm.endsc[v];
            let esc_v = &cm.esc[v];
            for j in 0..=w {
                let jrow = j * stride;
                for d in 0..=j {
                    let idx = jrow + d;
                    let i = j - d + 1;
                    match vst {
                        x if x == MP_ST => {
                            if j != w && d != j {
                                let escore = pair_sc(esc_v, dsq[i - 1], dsq[j + 1]);
                                let src = (j + sdr) * stride + (d + sd);
                                beta.j[mb + idx] =
                                    flogsum(beta.j[mb + idx], beta.j[vb + src] + endsc + escore);
                            }
                            if fill_l && j == w && d != j {
                                let escore = marg_sc(&lm[v], dsq[i - 1]);
                                let src = jrow + (d + sdl);
                                beta.l[mb + idx] =
                                    flogsum(beta.l[mb + idx], beta.l[vb + src] + endsc + escore);
                            }
                            if fill_r && i == 1 && j != w {
                                let escore = marg_sc(&rm[v], dsq[j + 1]);
                                let src = (j + sdr) * stride + (d + sdr);
                                beta.r[mb + idx] =
                                    flogsum(beta.r[mb + idx], beta.r[vb + src] + endsc + escore);
                            }
                        }
                        x if x == ML_ST || x == IL_ST => {
                            if d != j {
                                let escore = sing_sc(esc_v, dsq[i - 1]);
                                let srcj = jrow + (d + sd);
                                beta.j[mb + idx] =
                                    flogsum(beta.j[mb + idx], beta.j[vb + srcj] + endsc + escore);
                                if fill_l {
                                    let srcl = jrow + (d + sdl);
                                    beta.l[mb + idx] =
                                        flogsum(beta.l[mb + idx], beta.l[vb + srcl] + endsc + escore);
                                }
                            }
                            if fill_r && i == 1 {
                                beta.r[mb + idx] =
                                    flogsum(beta.r[mb + idx], beta.r[vb + idx] + endsc);
                            }
                        }
                        x if x == MR_ST || x == IR_ST => {
                            if j != w {
                                let escore = sing_sc(esc_v, dsq[j + 1]);
                                let srcj = (j + sdr) * stride + (d + sd);
                                beta.j[mb + idx] =
                                    flogsum(beta.j[mb + idx], beta.j[vb + srcj] + endsc + escore);
                                if fill_r {
                                    let srcr = (j + sdr) * stride + (d + sdr);
                                    beta.r[mb + idx] =
                                        flogsum(beta.r[mb + idx], beta.r[vb + srcr] + endsc + escore);
                                }
                            }
                            if fill_l && j == w {
                                beta.l[mb + idx] =
                                    flogsum(beta.l[mb + idx], beta.l[vb + idx] + endsc);
                            }
                        }
                        _ => {
                            // S, D, E
                            let src = (j + sdr) * stride + (d + sd);
                            beta.j[mb + idx] = flogsum(beta.j[mb + idx], beta.j[vb + src] + endsc);
                            if fill_l {
                                beta.l[mb + idx] = flogsum(beta.l[mb + idx], beta.l[vb + src] + endsc);
                            }
                            if fill_r {
                                beta.r[mb + idx] = flogsum(beta.r[mb + idx], beta.r[vb + src] + endsc);
                            }
                        }
                    }
                }
            }
        }
    }

    // C 7755-7763: EL->EL left-emitting transitions.
    if local_end {
        let mb = beta.base(m);
        let elself = cm.el_selfsc;
        for j in (1..=w).rev() {
            let jrow = j * stride;
            for d in (0..=(j - 1)).rev() {
                let idx = jrow + d;
                let idx1 = jrow + d + 1;
                beta.j[mb + idx] = flogsum(beta.j[mb + idx], beta.j[mb + idx1] + elself);
                if fill_l {
                    beta.l[mb + idx] = flogsum(beta.l[mb + idx], beta.l[mb + idx1] + elself);
                }
                if fill_r {
                    beta.r[mb + idx] = flogsum(beta.r[mb + idx], beta.r[mb + idx1] + elself);
                }
            }
        }
    }

    beta
}

/// C `cm_TrPosterior` (cm_dpalign_trunc.c:8682): post = ins + out - sc per plane/cell.
pub fn tr_posterior(cm: &CM, lp: i32, preset_mode: i8, ins: &TrMx, out: &TrMx) -> TrMx {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let (fill_l, fill_r, fill_t) = tr_fill_from_mode(preset_mode);
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let root = w * stride + w;
    let sc = match preset_mode {
        TRMODE_J => ins.j[root],
        TRMODE_L => ins.l[root],
        TRMODE_R => ins.r[root],
        _ => ins.t[root],
    };
    let mut post = TrMx::new(m, w, IMPOSSIBLE);

    // C 8712-8732: EL deck (v==M), if local ends on.
    if local_end {
        let mb = post.base(m);
        for j in 0..=w {
            for d in 0..=j {
                let idx = j * stride + d;
                post.j[mb + idx] = ins.j[mb + idx] + out.j[mb + idx] - sc;
                if fill_l {
                    post.l[mb + idx] = ins.l[mb + idx] + out.l[mb + idx] - sc;
                }
                if fill_r {
                    post.r[mb + idx] = ins.r[mb + idx] + out.r[mb + idx] - sc;
                }
            }
        }
    }
    // C 8735-8770: J (all v), then L, R, T.
    for v in (0..m).rev() {
        let vb = post.base(v);
        for j in 0..=w {
            for d in 0..=j {
                let idx = j * stride + d;
                post.j[vb + idx] = ins.j[vb + idx] + out.j[vb + idx] - sc;
            }
        }
    }
    if fill_l {
        for v in (0..m).rev() {
            let vb = post.base(v);
            for j in 0..=w {
                for d in 0..=j {
                    let idx = j * stride + d;
                    post.l[vb + idx] = ins.l[vb + idx] + out.l[vb + idx] - sc;
                }
            }
        }
    }
    if fill_r {
        for v in (0..m).rev() {
            let vb = post.base(v);
            for j in 0..=w {
                for d in 0..=j {
                    let idx = j * stride + d;
                    post.r[vb + idx] = ins.r[vb + idx] + out.r[vb + idx] - sc;
                }
            }
        }
    }
    if fill_t {
        for v in (0..m).rev() {
            if v == 0 || cm.sttype[v] as i32 == B_ST {
                let vb = post.base(v);
                for j in 0..=w {
                    for d in 0..=j {
                        let idx = j * stride + d;
                        post.t[vb + idx] = ins.t[vb + idx] + out.t[vb + idx] - sc;
                    }
                }
            }
        }
    }
    post
}

/// C `CM_TR_EMIT_MX`: per-state left/right emission posteriors, J and marginal planes.
pub struct TrEmitMx {
    pub jl_pp: Vec<Option<Vec<f32>>>, // 0..=M, index i (1..=L)
    pub ll_pp: Vec<Option<Vec<f32>>>,
    pub jr_pp: Vec<Option<Vec<f32>>>, // index j (1..=L)
    pub rr_pp: Vec<Option<Vec<f32>>>,
    pub sum: Vec<f32>, // 0..=L
}

/// C `sreEXP2(x)` = 2^x.
#[inline]
fn sre_exp2(x: f32) -> f64 {
    ((x as f64) * 0.69314718f64).exp()
}
/// C `Fscore2postcode` (cm_dpalign.c:5450) via FScore2Prob.
#[inline]
fn fscore2postcode(sc: f32) -> u8 {
    let p: f32 = if !not_impossible(sc) { 0.0 } else { sre_exp2(sc) as f32 };
    let pd = p as f64;
    if pd + 0.05 >= 1.0 {
        b'*'
    } else {
        (((pd + 0.05) * 10.0) as i32 as u8).wrapping_add(b'0')
    }
}

/// C `cm_TrEmitterPosterior` (cm_dpalign_trunc.c:9036): posterior prob that each state
/// emitted each residue (left/right, J and marginal), normalized so each residue's
/// total emission prob is 1.0, with MATP l_pp/r_pp combined across MP/ML/MR states.
pub fn tr_emitter_posterior(cm: &CM, lp: i32, preset_mode: i8, post: &TrMx) -> TrEmitMx {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let (fill_l, fill_r, _) = tr_fill_from_mode(preset_mode);
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let mut jl_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut ll_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut jr_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut rr_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            jl_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
            ll_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            jr_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
            rr_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
        }
    }
    if local_end {
        jl_pp[m] = Some(vec![IMPOSSIBLE; w + 1]);
        ll_pp[m] = Some(vec![IMPOSSIBLE; w + 1]);
        rr_pp[m] = Some(vec![IMPOSSIBLE; w + 1]);
    }

    // Step 1 (C 9063-9093): accumulate posterior over d for each state.
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        let sd = tr_state_delta(stt) as usize;
        let sdl = tr_state_left_delta(stt) as usize;
        let sdr = tr_state_right_delta(stt) as usize;
        let vb = post.base(v);
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            let jlp = jl_pp[v].as_mut().unwrap();
            for j in 1..=w {
                // C: i=j-sd+1, i-- per d in sd..=j  <=>  i = j-d+1 (>=1, no underflow).
                for d in sd..=j {
                    let i = j - d + 1;
                    jlp[i] = flogsum(jlp[i], post.j[vb + j * stride + d]);
                }
            }
            if fill_l {
                let llp = ll_pp[v].as_mut().unwrap();
                for j in 1..=w {
                    for d in sdl..=j {
                        let i = j - d + 1;
                        llp[i] = flogsum(llp[i], post.l[vb + j * stride + d]);
                    }
                }
            }
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            let jrp = jr_pp[v].as_mut().unwrap();
            for j in 1..=w {
                for d in sd..=j {
                    jrp[j] = flogsum(jrp[j], post.j[vb + j * stride + d]);
                }
            }
            if fill_r {
                let rrp = rr_pp[v].as_mut().unwrap();
                for j in 1..=w {
                    for d in sdr..=j {
                        rrp[j] = flogsum(rrp[j], post.r[vb + j * stride + d]);
                    }
                }
            }
        }
    }
    // EL contribution (C 9094-9118).
    if local_end {
        let mb = post.base(m);
        {
            let jlp = jl_pp[m].as_mut().unwrap();
            for j in 1..=w {
                for d in 1..=j {
                    let i = j - d + 1;
                    jlp[i] = flogsum(jlp[i], post.j[mb + j * stride + d]);
                }
            }
        }
        if fill_l {
            let llp = ll_pp[m].as_mut().unwrap();
            for j in 1..=w {
                for d in 1..=j {
                    let i = j - d + 1;
                    llp[i] = flogsum(llp[i], post.l[mb + j * stride + d]);
                }
            }
        }
        if fill_r {
            let rrp = rr_pp[m].as_mut().unwrap();
            for j in 1..=w {
                for d in 1..=j {
                    let i = j - d + 1;
                    rrp[i] = flogsum(rrp[i], post.r[mb + j * stride + d]);
                }
            }
        }
    }

    // Step 2 (C 9128-9176): normalize so each residue's total emission prob is 1.0.
    let mut sum = vec![IMPOSSIBLE; w + 1];
    for v in 0..=m {
        if let Some(jlp) = &jl_pp[v] {
            for i in 1..=w {
                sum[i] = flogsum(sum[i], jlp[i]);
            }
        }
        if fill_l {
            if let Some(llp) = &ll_pp[v] {
                for i in 1..=w {
                    sum[i] = flogsum(sum[i], llp[i]);
                }
            }
        }
        if let Some(jrp) = &jr_pp[v] {
            for j in 1..=w {
                sum[j] = flogsum(sum[j], jrp[j]);
            }
        }
        if fill_r {
            if let Some(rrp) = &rr_pp[v] {
                for j in 1..=w {
                    sum[j] = flogsum(sum[j], rrp[j]);
                }
            }
        }
    }
    for v in 0..=m {
        if let Some(jlp) = jl_pp[v].as_mut() {
            for i in 1..=w {
                jlp[i] -= sum[i];
            }
        }
        if fill_l {
            if let Some(llp) = ll_pp[v].as_mut() {
                for i in 1..=w {
                    llp[i] -= sum[i];
                }
            }
        }
        if let Some(jrp) = jr_pp[v].as_mut() {
            for j in 1..=w {
                jrp[j] -= sum[j];
            }
        }
        if fill_r {
            if let Some(rrp) = rr_pp[v].as_mut() {
                for j in 1..=w {
                    rrp[j] -= sum[j];
                }
            }
        }
    }

    // Step 3 (C 9186-9209): combine MATP_MP (v) with MATP_ML (v+1), MATP_MR (v+2).
    for v in 0..m {
        if cm.sttype[v] as i32 == MP_ST {
            for i in 1..=w {
                let a = jl_pp[v].as_ref().unwrap()[i];
                let b = jl_pp[v + 1].as_ref().unwrap()[i];
                let s = flogsum(a, b);
                jl_pp[v].as_mut().unwrap()[i] = s;
                jl_pp[v + 1].as_mut().unwrap()[i] = s;
            }
            if fill_l {
                for i in 1..=w {
                    let a = ll_pp[v].as_ref().unwrap()[i];
                    let b = ll_pp[v + 1].as_ref().unwrap()[i];
                    let s = flogsum(a, b);
                    ll_pp[v].as_mut().unwrap()[i] = s;
                    ll_pp[v + 1].as_mut().unwrap()[i] = s;
                }
            }
            for j in 1..=w {
                let a = jr_pp[v].as_ref().unwrap()[j];
                let b = jr_pp[v + 2].as_ref().unwrap()[j];
                let s = flogsum(a, b);
                jr_pp[v].as_mut().unwrap()[j] = s;
                jr_pp[v + 2].as_mut().unwrap()[j] = s;
            }
            if fill_r {
                for j in 1..=w {
                    let a = rr_pp[v].as_ref().unwrap()[j];
                    let b = rr_pp[v + 2].as_ref().unwrap()[j];
                    let s = flogsum(a, b);
                    rr_pp[v].as_mut().unwrap()[j] = s;
                    rr_pp[v + 2].as_mut().unwrap()[j] = s;
                }
            }
        }
    }

    TrEmitMx { jl_pp, ll_pp, jr_pp, rr_pp, sum }
}

/// C `cm_TrEmitterPosteriorHB` (cm_dpalign_trunc.c:9246): HMM-banded emitter posterior.
/// Uses raw-i storage (like [`tr_emitter_posterior`]) but reads the banded `post`
/// matrix and iterates the exact banded (j,d) order so FLogsum accumulation is
/// bit-identical to C; Step 2/3 use the i-band (imin/imax) and j-band (jmin/jmax).
pub fn tr_emitter_posterior_hb(
    cm: &CM,
    cp9b: &crate::cp9::CP9Bands,
    lp: i32,
    preset_mode: i8,
    post: &TrHbMx,
) -> TrEmitMx {
    let m = cm.m as usize;
    let w = lp as usize;
    let l = lp;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let imin = &cp9b.imin;
    let imax = &cp9b.imax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let (fill_l, fill_r, _) = tr_fill_from_mode(preset_mode);
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    let mut jl_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut ll_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut jr_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    let mut rr_pp: Vec<Option<Vec<f32>>> = vec![None; m + 1];
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            jl_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
            ll_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            jr_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
            rr_pp[v] = Some(vec![IMPOSSIBLE; w + 1]);
        }
    }
    if local_end {
        jl_pp[m] = Some(vec![IMPOSSIBLE; w + 1]);
        ll_pp[m] = Some(vec![IMPOSSIBLE; w + 1]);
        rr_pp[m] = Some(vec![IMPOSSIBLE; w + 1]);
    }

    // Step 1 (C 9285-9331): accumulate banded posterior over d for each state.
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        let njv = if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 };
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            if cp9b.jvalid[v] {
                let jlp = jl_pp[v].as_mut().unwrap();
                for jp_v in 0..njv {
                    let j = jmin[v] + jp_v as i32;
                    for dp in 0..=(hdmax[v][jp_v] - hdmin[v][jp_v]).max(-1) {
                        if hdmax[v][jp_v] < hdmin[v][jp_v] { break; }
                        let d = hdmin[v][jp_v] + dp;
                        let i = (j - d + 1) as usize;
                        jlp[i] = flogsum(jlp[i], post.j[v][jp_v][dp as usize]);
                    }
                }
            }
            if cp9b.lvalid[v] && fill_l {
                let llp = ll_pp[v].as_mut().unwrap();
                for jp_v in 0..njv {
                    let j = jmin[v] + jp_v as i32;
                    for dp in 0..=(hdmax[v][jp_v] - hdmin[v][jp_v]).max(-1) {
                        if hdmax[v][jp_v] < hdmin[v][jp_v] { break; }
                        let d = hdmin[v][jp_v] + dp;
                        let i = (j - d + 1) as usize;
                        llp[i] = flogsum(llp[i], post.l[v][jp_v][dp as usize]);
                    }
                }
            }
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            if cp9b.jvalid[v] {
                let jrp = jr_pp[v].as_mut().unwrap();
                for jp_v in 0..njv {
                    let j = (jmin[v] + jp_v as i32) as usize;
                    for dp in 0..=(hdmax[v][jp_v] - hdmin[v][jp_v]).max(-1) {
                        if hdmax[v][jp_v] < hdmin[v][jp_v] { break; }
                        jrp[j] = flogsum(jrp[j], post.j[v][jp_v][dp as usize]);
                    }
                }
            }
            if cp9b.rvalid[v] && fill_r {
                let rrp = rr_pp[v].as_mut().unwrap();
                for jp_v in 0..njv {
                    let j = (jmin[v] + jp_v as i32) as usize;
                    for dp in 0..=(hdmax[v][jp_v] - hdmin[v][jp_v]).max(-1) {
                        if hdmax[v][jp_v] < hdmin[v][jp_v] { break; }
                        rrp[j] = flogsum(rrp[j], post.r[v][jp_v][dp as usize]);
                    }
                }
            }
        }
    }
    // EL contribution (C 9335-9360), non-banded M deck.
    if local_end {
        if cp9b.jvalid[m] {
            let jlp = jl_pp[m].as_mut().unwrap();
            for j in 1..=w {
                for d in 1..=j {
                    let i = j - d + 1;
                    jlp[i] = flogsum(jlp[i], post.j[m][j][d]);
                }
            }
        }
        if fill_l && cp9b.lvalid[m] {
            let llp = ll_pp[m].as_mut().unwrap();
            for j in 1..=w {
                for d in 1..=j {
                    let i = j - d + 1;
                    llp[i] = flogsum(llp[i], post.l[m][j][d]);
                }
            }
        }
        if fill_r && cp9b.rvalid[m] {
            let rrp = rr_pp[m].as_mut().unwrap();
            for j in 1..=w {
                for d in 1..=j {
                    let i = j - d + 1;
                    rrp[i] = flogsum(rrp[i], post.r[m][j][d]);
                }
            }
        }
    }

    // Step 2 (C 9375-9440): normalize so each residue's total emission prob is 1.0.
    let mut sum = vec![IMPOSSIBLE; w + 1];
    // i32 inclusive ranges (empty when start>end, e.g. unset band jmax<0).
    let irange = |v: usize| -> std::ops::RangeInclusive<i32> {
        imin[v].max(1)..=imax[v].min(l)
    };
    let jrange = |v: usize| -> std::ops::RangeInclusive<i32> {
        jmin[v].max(1)..=jmax[v].min(l)
    };
    for v in 0..m {
        if cp9b.jvalid[v] {
            if let Some(jlp) = &jl_pp[v] {
                for i in irange(v) { let iu = i as usize; sum[iu] = flogsum(sum[iu], jlp[iu]); }
            }
        }
        if fill_l && cp9b.lvalid[v] {
            if let Some(llp) = &ll_pp[v] {
                for i in irange(v) { let iu = i as usize; sum[iu] = flogsum(sum[iu], llp[iu]); }
            }
        }
        if cp9b.jvalid[v] {
            if let Some(jrp) = &jr_pp[v] {
                for j in jrange(v) { let ju = j as usize; sum[ju] = flogsum(sum[ju], jrp[ju]); }
            }
        }
        if fill_r && cp9b.rvalid[v] {
            if let Some(rrp) = &rr_pp[v] {
                for j in jrange(v) { let ju = j as usize; sum[ju] = flogsum(sum[ju], rrp[ju]); }
            }
        }
    }
    // EL deck (non-banded, valid for Jl/Ll/Rr).
    if cp9b.jvalid[m] {
        if let Some(jlp) = &jl_pp[m] { for i in 1..=w { sum[i] = flogsum(sum[i], jlp[i]); } }
    }
    if fill_l && cp9b.lvalid[m] {
        if let Some(llp) = &ll_pp[m] { for i in 1..=w { sum[i] = flogsum(sum[i], llp[i]); } }
    }
    if fill_r && cp9b.rvalid[m] {
        if let Some(rrp) = &rr_pp[m] { for i in 1..=w { sum[i] = flogsum(sum[i], rrp[i]); } }
    }
    // normalize (subtract sum) over the same ranges.
    for v in 0..m {
        if cp9b.jvalid[v] {
            if let Some(jlp) = jl_pp[v].as_mut() { for i in irange(v) { let iu = i as usize; jlp[iu] -= sum[iu]; } }
        }
        if fill_l && cp9b.lvalid[v] {
            if let Some(llp) = ll_pp[v].as_mut() { for i in irange(v) { let iu = i as usize; llp[iu] -= sum[iu]; } }
        }
        if cp9b.jvalid[v] {
            if let Some(jrp) = jr_pp[v].as_mut() { for j in jrange(v) { let ju = j as usize; jrp[ju] -= sum[ju]; } }
        }
        if fill_r && cp9b.rvalid[v] {
            if let Some(rrp) = rr_pp[v].as_mut() { for j in jrange(v) { let ju = j as usize; rrp[ju] -= sum[ju]; } }
        }
    }
    if cp9b.jvalid[m] {
        if let Some(jlp) = jl_pp[m].as_mut() { for i in 1..=w { jlp[i] -= sum[i]; } }
    }
    if fill_l && cp9b.lvalid[m] {
        if let Some(llp) = ll_pp[m].as_mut() { for i in 1..=w { llp[i] -= sum[i]; } }
    }
    if fill_r && cp9b.rvalid[m] {
        if let Some(rrp) = rr_pp[m].as_mut() { for i in 1..=w { rrp[i] -= sum[i]; } }
    }

    // Step 3 (C 9490-9548): combine MATP_MP (v) with MATP_ML (v+1), MATP_MR (v+2),
    // over the i-band / j-band overlap.
    for v in 0..m {
        if cm.sttype[v] as i32 == MP_ST {
            if cp9b.jvalid[v] && imax[v] >= 1 && imax[v + 1] >= 1 {
                let in_ = imin[v].max(imin[v + 1]).max(1);
                let ix = imax[v].min(imax[v + 1]).min(l);
                for i in in_..=ix {
                    let iu = i as usize;
                    let a = jl_pp[v].as_ref().unwrap()[iu];
                    let b = jl_pp[v + 1].as_ref().unwrap()[iu];
                    let s = flogsum(a, b);
                    jl_pp[v].as_mut().unwrap()[iu] = s;
                    jl_pp[v + 1].as_mut().unwrap()[iu] = s;
                }
            }
            if cp9b.lvalid[v] && fill_l && imax[v] >= 1 && imax[v + 1] >= 1 {
                let in_ = imin[v].max(imin[v + 1]).max(1);
                let ix = imax[v].min(imax[v + 1]).min(l);
                for i in in_..=ix {
                    let iu = i as usize;
                    let a = ll_pp[v].as_ref().unwrap()[iu];
                    let b = ll_pp[v + 1].as_ref().unwrap()[iu];
                    let s = flogsum(a, b);
                    ll_pp[v].as_mut().unwrap()[iu] = s;
                    ll_pp[v + 1].as_mut().unwrap()[iu] = s;
                }
            }
            if cp9b.jvalid[v] && jmax[v] >= 1 && jmax[v + 2] >= 1 {
                let jn = jmin[v].max(jmin[v + 2]).max(1);
                let jx = jmax[v].min(jmax[v + 2]).min(l);
                for j in jn..=jx {
                    let ju = j as usize;
                    let a = jr_pp[v].as_ref().unwrap()[ju];
                    let b = jr_pp[v + 2].as_ref().unwrap()[ju];
                    let s = flogsum(a, b);
                    jr_pp[v].as_mut().unwrap()[ju] = s;
                    jr_pp[v + 2].as_mut().unwrap()[ju] = s;
                }
            }
            if cp9b.rvalid[v] && fill_r && jmax[v] >= 1 && jmax[v + 2] >= 1 {
                let jn = jmin[v].max(jmin[v + 2]).max(1);
                let jx = jmax[v].min(jmax[v + 2]).min(l);
                for j in jn..=jx {
                    let ju = j as usize;
                    let a = rr_pp[v].as_ref().unwrap()[ju];
                    let b = rr_pp[v + 2].as_ref().unwrap()[ju];
                    let s = flogsum(a, b);
                    rr_pp[v].as_mut().unwrap()[ju] = s;
                    rr_pp[v + 2].as_mut().unwrap()[ju] = s;
                }
            }
        }
    }

    TrEmitMx { jl_pp, ll_pp, jr_pp, rr_pp, sum }
}

/// C `NumReachableInserts` (cm.c:1274): number of insert states reachable from a
/// state, keyed by its stid. Used only for the special OptAcc child ordering.
fn num_reachable_inserts(stid: i32) -> i32 {
    match stid {
        x if x == MATL_ML => 1,
        x if x == MATL_D => 1,
        x if x == MATL_IL => 1,
        x if x == MATP_MP => 2,
        x if x == MATP_ML => 2,
        x if x == MATP_MR => 2,
        x if x == MATP_D => 2,
        x if x == MATP_IL => 2,
        x if x == MATP_IR => 1,
        x if x == MATR_MR => 1,
        x if x == MATR_D => 1,
        x if x == MATR_IR => 1,
        x if x == BIF_B => 0,
        x if x == BEGL_S => 0,
        x if x == BEGR_S => 1,
        x if x == BEGR_IL => 1,
        x if x == END_E => 0,
        x if x == ROOT_S => 2,
        x if x == ROOT_IL => 2,
        x if x == ROOT_IR => 1,
        x if x == EL => 0, // END_EL
        _ => panic!("num_reachable_inserts: bogus state id {}", stid),
    }
}

/// C `cm_TrOptAccAlign` (cm_dpalign_trunc.c:4331) + the OptAcc branch of
/// `cm_tr_alignT` (cm_dpalign_trunc.c:178). Non-banded truncated optimal-accuracy
/// alignment: max-accuracy DP over the J/L/R/T marginal planes, driven by the
/// pre-filled emission posteriors in `emit_mx`. Transitions carry NO score (only
/// emission posteriors and subtree combinations contribute, via FLogsum); child
/// selection is a max (`>`). `preset_mode` is the Inside-determined winning mode
/// (never UNKNOWN). Returns (parsetree, avg_pp). The traceback mirrors the CYK
/// traceback in `tr_cyk_align` but additionally inserts EL nodes on USED_EL
/// (reachable here because local ends may be on).
#[allow(clippy::too_many_arguments)]
pub fn tr_optacc_align(
    cm: &CM,
    lp: i32,
    preset_mode: i8,
    pass_idx: i32,
    trp: &TrPenalties,
    emit_mx: &TrEmitMx,
) -> (Parsetree, f32) {
    let m = cm.m as usize;
    let w = lp as usize;
    let stride = w + 1;
    let ncell = stride * stride;
    let (fill_l, fill_r, fill_t) = tr_fill_from_mode(preset_mode);
    let have_el = cm.flags & CMH_LOCAL_END != 0;
    // C: trpenalty = have_el ? l_ptyAA[pty_idx][v] : g_ptyAA[pty_idx][v] (4877).
    let pty = trp.pty_slice(pass_idx, have_el);

    // Per-state decks; index [v] size m+1 for alpha (index m = EL deck), size m for
    // shadows (state 0 handled specially in traceback but still allocated for dzero).
    let mut jalpha: Vec<Vec<f32>> = vec![Vec::new(); m + 1];
    let mut lalpha: Vec<Vec<f32>> = vec![Vec::new(); m + 1];
    let mut ralpha: Vec<Vec<f32>> = vec![Vec::new(); m + 1];
    let mut talpha: Vec<Vec<f32>> = vec![Vec::new(); m]; // B only
    let mut jyshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut lyshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut ryshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut jkshad: Vec<Vec<i32>> = vec![Vec::new(); m]; // B only
    let mut lkshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut rkshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut tkshad: Vec<Vec<i32>> = vec![Vec::new(); m];
    let mut lkmode: Vec<Vec<i8>> = vec![Vec::new(); m];
    let mut rkmode: Vec<Vec<i8>> = vec![Vec::new(); m];

    // Allocate/initialize all decks (C: FSet IMPOSSIBLE, shadows USED_EL / 0 / TRMODE_J).
    for v in 0..=m {
        jalpha[v] = vec![IMPOSSIBLE; ncell];
        if fill_l {
            lalpha[v] = vec![IMPOSSIBLE; ncell];
        }
        if fill_r {
            ralpha[v] = vec![IMPOSSIBLE; ncell];
        }
    }
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        jyshad[v] = vec![USED_EL; ncell];
        if fill_l {
            lyshad[v] = vec![USED_EL; ncell];
        }
        if fill_r {
            ryshad[v] = vec![USED_EL; ncell];
        }
        if stt == B_ST {
            jkshad[v] = vec![0i32; ncell];
            if fill_l {
                lkshad[v] = vec![0i32; ncell];
                lkmode[v] = vec![TRMODE_J; ncell];
            }
            if fill_r {
                rkshad[v] = vec![0i32; ncell];
                rkmode[v] = vec![TRMODE_J; ncell];
            }
            if fill_t {
                tkshad[v] = vec![0i32; ncell];
                talpha[v] = vec![IMPOSSIBLE; ncell];
            }
        }
    }

    // C: cm_InitializeOptAccShadowDZero on Jyshadow (d==0 zero-length parse paths).
    crate::cm_dpalign::cm_init_optacc_shadow_dzero(cm, &mut jyshad, lp, stride);

    // EL deck (index m) init from emit posteriors, if local ends are on (C 4419-4437).
    if have_el {
        for j in 0..=w {
            if let Some(jl) = &emit_mx.jl_pp[m] {
                jalpha[m][j * stride] = jl[0];
            }
            if fill_l {
                if let Some(ll) = &emit_mx.ll_pp[m] {
                    lalpha[m][j * stride] = ll[0];
                }
            }
            if fill_r {
                if let Some(rr) = &emit_mx.rr_pp[m] {
                    ralpha[m][j * stride] = rr[0];
                }
            }
            if let Some(jl) = &emit_mx.jl_pp[m] {
                for d in 1..=j {
                    let i = j - d + 1;
                    jalpha[m][j * stride + d] = flogsum(jalpha[m][j * stride + d - 1], jl[i]);
                }
            }
            if fill_l {
                if let Some(ll) = &emit_mx.ll_pp[m] {
                    for d in 1..=j {
                        let i = j - d + 1;
                        lalpha[m][j * stride + d] = flogsum(lalpha[m][j * stride + d - 1], ll[i]);
                    }
                }
            }
            if fill_r {
                if let Some(rr) = &emit_mx.rr_pp[m] {
                    for d in 1..=j {
                        let i = j - d + 1;
                        ralpha[m][j * stride + d] = flogsum(ralpha[m][j * stride + d - 1], rr[i]);
                    }
                }
            }
        }
    }

    let mut best = IMPOSSIBLE;
    let mut b = 0i32;

    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let nins_v = num_reachable_inserts(cm.stid[v] as i32) as usize;
        let sd = tr_state_delta(stt) as usize;
        let sdl = tr_state_left_delta(stt) as usize;
        let sdr = tr_state_right_delta(stt) as usize;

        let mut ja = std::mem::take(&mut jalpha[v]);
        let mut la = if fill_l { std::mem::take(&mut lalpha[v]) } else { Vec::new() };
        let mut ra = if fill_r { std::mem::take(&mut ralpha[v]) } else { Vec::new() };
        let mut jy = std::mem::take(&mut jyshad[v]);
        let mut ly = if fill_l { std::mem::take(&mut lyshad[v]) } else { Vec::new() };
        let mut ry = if fill_r { std::mem::take(&mut ryshad[v]) } else { Vec::new() };
        let mut ta = if stt == B_ST && fill_t { std::mem::take(&mut talpha[v]) } else { Vec::new() };
        let mut jk = if stt == B_ST { std::mem::take(&mut jkshad[v]) } else { Vec::new() };
        let mut lk = if stt == B_ST && fill_l { std::mem::take(&mut lkshad[v]) } else { Vec::new() };
        let mut rk = if stt == B_ST && fill_r { std::mem::take(&mut rkshad[v]) } else { Vec::new() };
        let mut tk = if stt == B_ST && fill_t { std::mem::take(&mut tkshad[v]) } else { Vec::new() };
        let mut lkm = if stt == B_ST && fill_l { std::mem::take(&mut lkmode[v]) } else { Vec::new() };
        let mut rkm = if stt == B_ST && fill_r { std::mem::take(&mut rkmode[v]) } else { Vec::new() };

        // EL re-init (copy from saved EL deck), C 4447-4453.
        if have_el && not_impossible(cm.endsc[v]) {
            for j in 0..=w {
                for d in sd..=j {
                    ja[j * stride + d] = jalpha[m][(j - sdr) * stride + (d - sd)];
                }
                if fill_l {
                    for d in sdl..=j {
                        la[j * stride + d] = lalpha[m][j * stride + (d - sdl)];
                    }
                }
                if fill_r {
                    for d in sdr..=j {
                        ra[j * stride + d] = ralpha[m][(j - sdr) * stride + (d - sdr)];
                    }
                }
            }
        }

        if stt == IL_ST || stt == ML_ST {
            if !state_is_detached(cm, v) {
                let ryoffset0 = if stt == IL_ST { 1 } else { 0 };
                let jlp = emit_mx.jl_pp[v].as_ref().unwrap();
                let llp = emit_mx.ll_pp[v].as_ref();
                for j in 0..=w {
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let i = j - d + 1;
                        let d_sd = d - 1;
                        for yctr in 0..cnum {
                            let yoffset = (yctr + nins_v) % cnum;
                            let y = cfirst + yoffset;
                            let jc = if y == v { ja[j * stride + d_sd] } else { jalpha[y][j * stride + d_sd] };
                            if jc > ja[idx] {
                                ja[idx] = jc;
                                jy[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                            }
                            if fill_l {
                                let lc = if y == v { la[j * stride + d_sd] } else { lalpha[y][j * stride + d_sd] };
                                if lc > la[idx] {
                                    la[idx] = lc;
                                    ly[idx] = yoffset as i32 + TRMODE_L_OFFSET;
                                }
                            }
                        }
                        ja[idx] = flogsum(ja[idx], jlp[i]);
                        if ja[idx] < IMPOSSIBLE {
                            ja[idx] = IMPOSSIBLE;
                        }
                        if !have_el && jy[idx] == USED_EL && d > 1 {
                            ja[idx] = IMPOSSIBLE;
                        }
                        if fill_l {
                            let llp = llp.unwrap();
                            if d >= 2 {
                                la[idx] = flogsum(la[idx], llp[i]);
                                if !have_el && ly[idx] == USED_EL {
                                    la[idx] = IMPOSSIBLE;
                                }
                            } else {
                                la[idx] = llp[i];
                                ly[idx] = USED_TRUNC_END;
                            }
                            if la[idx] < IMPOSSIBLE {
                                la[idx] = IMPOSSIBLE;
                            }
                        }
                        if fill_r {
                            for yctr in ryoffset0..cnum {
                                let yoffset = (yctr + nins_v - ryoffset0) % cnum;
                                let y = cfirst + yoffset;
                                let jc = if y == v { ja[j * stride + d] } else { jalpha[y][j * stride + d] };
                                if jc > ra[idx] {
                                    ra[idx] = jc;
                                    ry[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                                }
                                let rc = if y == v { ra[j * stride + d] } else { ralpha[y][j * stride + d] };
                                if rc > ra[idx] {
                                    ra[idx] = rc;
                                    ry[idx] = yoffset as i32 + TRMODE_R_OFFSET;
                                }
                            }
                            if ra[idx] < IMPOSSIBLE {
                                ra[idx] = IMPOSSIBLE;
                            }
                        }
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            if !state_is_detached(cm, v) {
                let lyoffset0 = if stt == IR_ST { 1 } else { 0 };
                let jrp = emit_mx.jr_pp[v].as_ref().unwrap();
                let rrp = emit_mx.rr_pp[v].as_ref();
                for j in 1..=w {
                    let j_sdr = j - 1;
                    for d in 1..=j {
                        let idx = j * stride + d;
                        let d_sd = d - 1;
                        for yctr in 0..cnum {
                            let yoffset = (yctr + nins_v) % cnum;
                            let y = cfirst + yoffset;
                            let jc = if y == v { ja[j_sdr * stride + d_sd] } else { jalpha[y][j_sdr * stride + d_sd] };
                            if jc > ja[idx] {
                                ja[idx] = jc;
                                jy[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                            }
                            if fill_r {
                                let rc = if y == v { ra[j_sdr * stride + d_sd] } else { ralpha[y][j_sdr * stride + d_sd] };
                                if rc > ra[idx] {
                                    ra[idx] = rc;
                                    ry[idx] = yoffset as i32 + TRMODE_R_OFFSET;
                                }
                            }
                        }
                        ja[idx] = flogsum(ja[idx], jrp[j]);
                        if ja[idx] < IMPOSSIBLE {
                            ja[idx] = IMPOSSIBLE;
                        }
                        if !have_el && jy[idx] == USED_EL && d > 1 {
                            ja[idx] = IMPOSSIBLE;
                        }
                        if fill_r {
                            let rrp = rrp.unwrap();
                            if d >= 2 {
                                ra[idx] = flogsum(ra[idx], rrp[j]);
                                if !have_el && ry[idx] == USED_EL {
                                    ra[idx] = IMPOSSIBLE;
                                }
                            } else {
                                ra[idx] = rrp[j];
                                ry[idx] = USED_TRUNC_END;
                            }
                            if ra[idx] < IMPOSSIBLE {
                                ra[idx] = IMPOSSIBLE;
                            }
                        }
                        if fill_l {
                            for yctr in lyoffset0..cnum {
                                let yoffset = (yctr + nins_v - lyoffset0) % cnum;
                                let y = cfirst + yoffset;
                                let jc = if y == v { ja[j * stride + d] } else { jalpha[y][j * stride + d] };
                                if jc > la[idx] {
                                    la[idx] = jc;
                                    ly[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                                }
                                let lc = if y == v { la[j * stride + d] } else { lalpha[y][j * stride + d] };
                                if lc > la[idx] {
                                    la[idx] = lc;
                                    ly[idx] = yoffset as i32 + TRMODE_L_OFFSET;
                                }
                            }
                            if la[idx] < IMPOSSIBLE {
                                la[idx] = IMPOSSIBLE;
                            }
                        }
                    }
                }
            }
        } else if stt == MP_ST {
            for yctr in 0..cnum {
                let yoffset = (yctr + nins_v) % cnum;
                let y = cfirst + yoffset;
                for j in 1..=w {
                    let j_sdr = j - 1;
                    for d in 2..=j {
                        let d_sd = d - 2;
                        let idx = j * stride + d;
                        let jc = jalpha[y][j_sdr * stride + d_sd];
                        if jc > ja[idx] {
                            ja[idx] = jc;
                            jy[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                        }
                    }
                    if fill_l {
                        for d in 1..=j {
                            let d_sdl = d - 1;
                            let idx = j * stride + d;
                            let jc = jalpha[y][j * stride + d_sdl];
                            if jc > la[idx] {
                                la[idx] = jc;
                                ly[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                            }
                            let lc = lalpha[y][j * stride + d_sdl];
                            if lc > la[idx] {
                                la[idx] = lc;
                                ly[idx] = yoffset as i32 + TRMODE_L_OFFSET;
                            }
                        }
                    }
                    if fill_r {
                        for d in 1..=j {
                            let d_sdr = d - 1;
                            let idx = j * stride + d;
                            let jc = jalpha[y][j_sdr * stride + d_sdr];
                            if jc > ra[idx] {
                                ra[idx] = jc;
                                ry[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                            }
                            let rc = ralpha[y][j_sdr * stride + d_sdr];
                            if rc > ra[idx] {
                                ra[idx] = rc;
                                ry[idx] = yoffset as i32 + TRMODE_R_OFFSET;
                            }
                        }
                    }
                }
            }
            // emission (C 4671-4696)
            let jlp = emit_mx.jl_pp[v].as_ref().unwrap();
            let jrp = emit_mx.jr_pp[v].as_ref().unwrap();
            let llp = emit_mx.ll_pp[v].as_ref();
            let rrp = emit_mx.rr_pp[v].as_ref();
            for j in 0..=w {
                ja[j * stride + 1] = IMPOSSIBLE;
                if fill_l && j >= 1 {
                    let i = j; // j-1+1
                    la[j * stride + 1] = llp.unwrap()[i];
                    ly[j * stride + 1] = USED_TRUNC_END;
                }
                if fill_r && j >= 1 {
                    ra[j * stride + 1] = rrp.unwrap()[j];
                    ry[j * stride + 1] = USED_TRUNC_END;
                }
                for d in 2..=j {
                    let i = j - d + 1;
                    let idx = j * stride + d;
                    ja[idx] = flogsum(ja[idx], flogsum(jlp[i], jrp[j]));
                }
                if fill_l {
                    for d in 2..=j {
                        let i = j - d + 1;
                        la[j * stride + d] = flogsum(la[j * stride + d], llp.unwrap()[i]);
                    }
                }
                if fill_r {
                    for d in 2..=j {
                        ra[j * stride + d] = flogsum(ra[j * stride + d], rrp.unwrap()[j]);
                    }
                }
            }
            // clamp (C 4699-4703)
            for j in 0..=w {
                for d in 1..=j {
                    let idx = j * stride + d;
                    if ja[idx] < IMPOSSIBLE {
                        ja[idx] = IMPOSSIBLE;
                    }
                }
                if fill_l {
                    for d in 1..=j {
                        let idx = j * stride + d;
                        if la[idx] < IMPOSSIBLE {
                            la[idx] = IMPOSSIBLE;
                        }
                    }
                }
                if fill_r {
                    for d in 1..=j {
                        let idx = j * stride + d;
                        if ra[idx] < IMPOSSIBLE {
                            ra[idx] = IMPOSSIBLE;
                        }
                    }
                }
            }
            // disallow illegal EL emissions if local ends off (C 4707-4723)
            if !have_el {
                for j in 0..=w {
                    for d in 3..=j {
                        let idx = j * stride + d;
                        if jy[idx] == USED_EL {
                            ja[idx] = IMPOSSIBLE;
                        }
                    }
                    if fill_l {
                        for d in 2..=j {
                            let idx = j * stride + d;
                            if ly[idx] == USED_EL {
                                la[idx] = IMPOSSIBLE;
                            }
                        }
                    }
                    if fill_r {
                        for d in 2..=j {
                            let idx = j * stride + d;
                            if ry[idx] == USED_EL {
                                ra[idx] = IMPOSSIBLE;
                            }
                        }
                    }
                }
            }
        } else if stt != B_ST {
            // D, S, (E: cnum==0 so nothing happens) states
            for yctr in 0..cnum {
                let yoffset = (yctr + nins_v) % cnum;
                let y = cfirst + yoffset;
                for j in 0..=w {
                    for d in 0..=j {
                        let idx = j * stride + d;
                        let jc = jalpha[y][j * stride + d];
                        if jc > ja[idx] {
                            ja[idx] = jc;
                            jy[idx] = yoffset as i32 + TRMODE_J_OFFSET;
                        }
                    }
                    if fill_l {
                        for d in 0..=j {
                            let idx = j * stride + d;
                            let lc = lalpha[y][j * stride + d];
                            if lc > la[idx] {
                                la[idx] = lc;
                                ly[idx] = yoffset as i32 + TRMODE_L_OFFSET;
                            }
                        }
                    }
                    if fill_r {
                        for d in 0..=j {
                            let idx = j * stride + d;
                            let rc = ralpha[y][j * stride + d];
                            if rc > ra[idx] {
                                ra[idx] = rc;
                                ry[idx] = yoffset as i32 + TRMODE_R_OFFSET;
                            }
                        }
                    }
                    if fill_l {
                        la[j * stride] = IMPOSSIBLE;
                    }
                    if fill_r {
                        ra[j * stride] = IMPOSSIBLE;
                    }
                    if stt == S_ST {
                        if fill_l {
                            ly[j * stride] = USED_TRUNC_END;
                        }
                        if fill_r {
                            ry[j * stride] = USED_TRUNC_END;
                        }
                    }
                }
            }
        } else {
            // B_st. NOTE: faithful reproduction of C 4787-4856, including the L-block
            // quirks (score uses Jalpha[z] and compares against ja, not la).
            let y = cfirst; // BEGL_S (left)
            let z = cnum; // BEGR_S (right) — for B, cnum[v] holds right child state index
            for j in 0..=w {
                for d in 0..=j {
                    let idx = j * stride + d;
                    for k in 0..=d {
                        let left = jalpha[y][(j - k) * stride + (d - k)];
                        let right = jalpha[z][j * stride + k];
                        if (not_impossible(left) || d == k) && (not_impossible(right) || k == 0) {
                            let sc = flogsum(left, right);
                            if sc > ja[idx] {
                                ja[idx] = sc;
                                jk[idx] = k as i32;
                            }
                        }
                    }
                    if fill_l {
                        for k in 0..=d {
                            let left = jalpha[y][(j - k) * stride + (d - k)];
                            let lz = lalpha[z][j * stride + k];
                            let right = jalpha[z][j * stride + k]; // C 4801 uses Jalpha[z]
                            if (not_impossible(left) || d == k) && (not_impossible(lz) || k == 0) {
                                let sc = flogsum(left, right);
                                if sc > ja[idx] {
                                    // C 4801 compares against Jalpha[v][j][d]
                                    la[idx] = sc;
                                    lk[idx] = k as i32;
                                    lkm[idx] = TRMODE_J;
                                }
                            }
                        }
                    }
                    if fill_r {
                        for k in 0..=d {
                            let left = ralpha[y][(j - k) * stride + (d - k)];
                            let right = jalpha[z][j * stride + k];
                            if (not_impossible(left) || d == k) && (not_impossible(right) || k == 0) {
                                let sc = flogsum(left, right);
                                if sc > ra[idx] {
                                    ra[idx] = sc;
                                    rk[idx] = k as i32;
                                    rkm[idx] = TRMODE_J;
                                }
                            }
                        }
                    }
                    if fill_t {
                        for k in 1..d {
                            let left = ralpha[y][(j - k) * stride + (d - k)];
                            let right = lalpha[z][j * stride + k];
                            if not_impossible(left) && not_impossible(right) {
                                let sc = flogsum(left, right);
                                if sc > ta[idx] {
                                    ta[idx] = sc;
                                    tk[idx] = k as i32;
                                }
                            }
                        }
                    }
                    // special case 1: k==0, full seq on left (C 4831-4842)
                    if fill_l {
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
                    }
                    // special case 2: k==d, full seq on right (C 4844-4855)
                    if fill_r {
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
        }

        // ROOT truncated-begin update (C 4877-4905): OptAcc tracks a single best b
        // for the given preset_mode; the penalty is only tested for NOT_IMPOSSIBLE.
        if not_impossible(pty[v]) {
            let root = w * stride + w;
            let cand = match preset_mode {
                TRMODE_J => Some(ja[root]),
                TRMODE_L => Some(la[root]),
                TRMODE_R => Some(ra[root]),
                _ => {
                    if stt == B_ST {
                        Some(ta[root])
                    } else {
                        None
                    }
                }
            };
            if let Some(c) = cand {
                if c > best {
                    best = c;
                    b = v as i32;
                }
            }
        }

        jalpha[v] = ja;
        if fill_l {
            lalpha[v] = la;
        }
        if fill_r {
            ralpha[v] = ra;
        }
        jyshad[v] = jy;
        if fill_l {
            lyshad[v] = ly;
        }
        if fill_r {
            ryshad[v] = ry;
        }
        if stt == B_ST {
            jkshad[v] = jk;
            if fill_l {
                lkshad[v] = lk;
                lkmode[v] = lkm;
            }
            if fill_r {
                rkshad[v] = rk;
                rkmode[v] = rkm;
            }
            if fill_t {
                tkshad[v] = tk;
                talpha[v] = ta;
            }
        }
    }

    let pp = (sre_exp2(best) / (w as f32) as f64) as f32;

    // ---- traceback (OptAcc branch of cm_tr_alignT); mirrors tr_cyk_align's
    // traceback but inserts an EL node on USED_EL (local ends may be on). ----
    let mut mode = preset_mode;
    let mut tr = Parsetree::new(w + 4);
    tr.is_std = false; // C cm_tr_alignT (optacc): truncated parse
    tr.pass_idx = pass_idx;
    // C cm_dpalign_trunc.c:233: tr->trpenalty = (local?l:g)_ptyAA[pty_idx][b].
    tr.trpenalty = pty[b as usize];
    tr.add_node_mode(1, lp, 0, -1, -1, -1, mode);
    let mut pda_i: Vec<i32> = Vec::new();
    let mut pda_c: Vec<i8> = Vec::new();
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = lp;
    let mut d: i32 = lp;

    loop {
        let vu = v as usize;
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
                _ => tkshad[vu][didx],
            };
            let prvmode = mode;
            let rmode = match prvmode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => TRMODE_L,
                TRMODE_R => rkmode[vu][didx],
                _ => TRMODE_L,
            };
            let bpar = tr.n - 1;
            pda_c.push(rmode);
            pda_i.push(j);
            pda_i.push(k);
            pda_i.push(bpar);
            let lmode = match prvmode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => lkmode[vu][didx],
                TRMODE_R => TRMODE_R,
                _ => TRMODE_R,
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

        let yoffset_raw = if v == 0 {
            USED_TRUNC_BEGIN
        } else {
            match mode {
                TRMODE_J => jyshad[vu][didx],
                TRMODE_L => lyshad[vu][didx],
                TRMODE_R => ryshad[vu][didx],
                _ => USED_TRUNC_BEGIN,
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
            _ => {}
        }
        d = j - i + 1;

        if yoffset == USED_EL || yoffset == USED_TRUNC_END {
            if yoffset == USED_EL {
                let idx = tr.add_node_mode(i, j, cm.m, -1, -1, tr.n - 1, mode);
                tr.nxtl[(tr.n - 2) as usize] = idx;
            }
            v = cm.m;
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

    (tr, pp)
}

/// HMM-banded truncated optimal-accuracy alignment. Faithful port of C
/// `cm_TrOptAccAlignHB` (cm_dpalign_trunc.c:4968): max-DP over emission
/// posteriors (child selection via `>`, emissions accumulated via FLogsum) on
/// banded `[v][jp][dp]` decks, with a NON-banded EL deck (index m) seeded from
/// `emit_mx`. Followed by the OptAcc branch of `cm_tr_alignT_hb`
/// (cm_dpalign_trunc.c:471). `emit_mx` is raw-i/j indexed (see
/// `tr_emitter_posterior_hb`), so wherever C uses `Jl_pp[v][ip_v]` /
/// `Jr_pp[v][jp_v]` we index `jl_pp[v][i]` / `jr_pp[v][j]` with raw i/j.
/// Returns (parsetree, avg posterior prob of emitted residues).
pub fn tr_optacc_align_hb(
    cm: &CM,
    cp9b: &crate::cp9::CP9Bands,
    lp: i32,
    pass_idx: i32,
    preset_mode: i8,
    trp: &TrPenalties,
    emit_mx: &TrEmitMx,
) -> (Parsetree, f32) {
    let m = cm.m as usize;
    let l = lp;
    let w = lp as usize;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let (fill_l, fill_r, fill_t) = tr_fill_from_mode(preset_mode);
    let have_el = cm.flags & CMH_LOCAL_END != 0;
    // C: trpenalty = have_el ? l_ptyAA[pty_idx][v] : g_ptyAA[pty_idx][v] (6008).
    let pty = trp.pty_slice(pass_idx, have_el);

    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    let njv = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };

    // ---- allocate banded {J,L,R,T} decks (0..m) + shadows; index m = non-banded EL ----
    let mut jalpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut lalpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut ralpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m + 1);
    let mut talpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m); // B only (no EL, no ROOT)
    let mut jysh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut lysh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut rysh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut jksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut lksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut rksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut tksh: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut lkmode: Vec<Vec<Vec<i8>>> = Vec::with_capacity(m);
    let mut rkmode: Vec<Vec<Vec<i8>>> = Vec::with_capacity(m);
    for v in 0..m {
        let nj = njv(v);
        let is_b = cm.sttype[v] as i32 == B_ST;
        let mut jd = Vec::with_capacity(nj);
        let mut ld = Vec::with_capacity(nj);
        let mut rd = Vec::with_capacity(nj);
        let mut td = Vec::with_capacity(nj);
        let mut jys = Vec::with_capacity(nj);
        let mut lys = Vec::with_capacity(nj);
        let mut rys = Vec::with_capacity(nj);
        let mut jks = Vec::with_capacity(nj);
        let mut lks = Vec::with_capacity(nj);
        let mut rks = Vec::with_capacity(nj);
        let mut tks = Vec::with_capacity(nj);
        let mut lkm = Vec::with_capacity(nj);
        let mut rkm = Vec::with_capacity(nj);
        for jp in 0..nj {
            let ww = (hdmax[v][jp] - hdmin[v][jp] + 1).max(0) as usize;
            jd.push(vec![IMPOSSIBLE; ww]);
            if fill_l { ld.push(vec![IMPOSSIBLE; ww]); }
            if fill_r { rd.push(vec![IMPOSSIBLE; ww]); }
            jys.push(vec![USED_EL; ww]);
            if fill_l { lys.push(vec![USED_EL; ww]); }
            if fill_r { rys.push(vec![USED_EL; ww]); }
            if is_b && fill_t { td.push(vec![IMPOSSIBLE; ww]); }
            if is_b {
                jks.push(vec![0i32; ww]);
                if fill_l { lks.push(vec![0i32; ww]); lkm.push(vec![TRMODE_J; ww]); }
                if fill_r { rks.push(vec![0i32; ww]); rkm.push(vec![TRMODE_J; ww]); }
                if fill_t { tks.push(vec![0i32; ww]); }
            }
        }
        jalpha.push(jd); lalpha.push(ld); ralpha.push(rd); talpha.push(td);
        jysh.push(jys); lysh.push(lys); rysh.push(rys);
        jksh.push(jks); lksh.push(lks); rksh.push(rks); tksh.push(tks);
        lkmode.push(lkm); rkmode.push(rkm);
    }
    // EL deck (v == cm.M): non-banded triangular, rows j=0..=L width j+1 (d=0..=j).
    let el_rows = |valid: bool| -> Vec<Vec<f32>> {
        if valid { (0..=w).map(|j| vec![IMPOSSIBLE; j + 1]).collect() } else { Vec::new() }
    };
    jalpha.push(el_rows(cp9b.jvalid[m]));
    lalpha.push(el_rows(fill_l && cp9b.lvalid[m]));
    ralpha.push(el_rows(fill_r && cp9b.rvalid[m]));

    // C cm_InitializeOptAccShadowDZeroHB on Jyshadow (d==0 zero-length parse paths).
    crate::cm_dpalign::cm_init_optacc_shadow_dzero_hb(cm, cp9b, &mut jysh, l);

    // EL deck init from emit posteriors (C 5100-5130), non-banded triangular.
    if have_el {
        let do_j_m = cp9b.jvalid[m] && emit_mx.jl_pp[m].is_some();
        let do_l_m = cp9b.lvalid[m] && emit_mx.ll_pp[m].is_some() && fill_l;
        let do_r_m = cp9b.rvalid[m] && emit_mx.rr_pp[m].is_some() && fill_r;
        for j in 0..=w {
            if do_j_m { jalpha[m][j][0] = emit_mx.jl_pp[m].as_ref().unwrap()[0]; }
            if do_l_m { lalpha[m][j][0] = emit_mx.ll_pp[m].as_ref().unwrap()[0]; }
            if do_r_m { ralpha[m][j][0] = emit_mx.rr_pp[m].as_ref().unwrap()[0]; }
            if do_j_m {
                let jl = emit_mx.jl_pp[m].as_ref().unwrap();
                for d in 1..=j { let i = j - d + 1; jalpha[m][j][d] = flogsum(jalpha[m][j][d - 1], jl[i]); }
            }
            if do_l_m {
                let ll = emit_mx.ll_pp[m].as_ref().unwrap();
                for d in 1..=j { let i = j - d + 1; lalpha[m][j][d] = flogsum(lalpha[m][j][d - 1], ll[i]); }
            }
            if do_r_m {
                let rr = emit_mx.rr_pp[m].as_ref().unwrap();
                for d in 1..=j { let i = j - d + 1; ralpha[m][j][d] = flogsum(ralpha[m][j][d - 1], rr[i]); }
            }
        }
    }

    let mut best = IMPOSSIBLE;
    let mut b = 0i32;

    // ---- main recursion v = M-1 downto 1 (C 5133) ----
    for v in (1..m).rev() {
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let nins_v = num_reachable_inserts(cm.stid[v] as i32) as usize;
        let sd = tr_state_delta(stt);
        let sdl = tr_state_left_delta(stt);
        let sdr = tr_state_right_delta(stt);
        let is_b = stt == B_ST;
        let do_j_v = cp9b.jvalid[v];
        let do_l_v = cp9b.lvalid[v] && fill_l;
        let do_r_v = cp9b.rvalid[v] && fill_r;
        let do_t_v = cp9b.tvalid[v] && fill_t;

        let mut ja = std::mem::take(&mut jalpha[v]);
        let mut la = if fill_l { std::mem::take(&mut lalpha[v]) } else { Vec::new() };
        let mut ra = if fill_r { std::mem::take(&mut ralpha[v]) } else { Vec::new() };
        let mut jy = std::mem::take(&mut jysh[v]);
        let mut ly = if fill_l { std::mem::take(&mut lysh[v]) } else { Vec::new() };
        let mut ry = if fill_r { std::mem::take(&mut rysh[v]) } else { Vec::new() };
        let mut ta = if is_b && fill_t { std::mem::take(&mut talpha[v]) } else { Vec::new() };
        let mut jk = if is_b { std::mem::take(&mut jksh[v]) } else { Vec::new() };
        let mut lk = if is_b && fill_l { std::mem::take(&mut lksh[v]) } else { Vec::new() };
        let mut rk = if is_b && fill_r { std::mem::take(&mut rksh[v]) } else { Vec::new() };
        let mut tk = if is_b && fill_t { std::mem::take(&mut tksh[v]) } else { Vec::new() };
        let mut lkm = if is_b && fill_l { std::mem::take(&mut lkmode[v]) } else { Vec::new() };
        let mut rkm = if is_b && fill_r { std::mem::take(&mut rkmode[v]) } else { Vec::new() };

        // re-init if we can do a local end from v, copy from saved EL deck (C 5106-5130).
        // Signed-index guard: C accesses Jalpha[M][j-sdr][d-sd] unguarded; in practice
        // d>=sd and j>=sdr hold whenever endsc[v] is valid (else C would read OOB).
        if have_el && not_impossible(cm.endsc[v]) {
            if do_j_v && cp9b.jvalid[m] {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let js = j - sdr;
                    for dp in 0..ja[jp_v].len() {
                        let d = hdmin[v][jp_v] + dp as i32;
                        let ds = d - sd;
                        if js >= 0 && ds >= 0 { ja[jp_v][dp] = jalpha[m][js as usize][ds as usize]; }
                    }
                }
            }
            if do_l_v && cp9b.lvalid[m] {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    for dp in 0..la[jp_v].len() {
                        let d = hdmin[v][jp_v] + dp as i32;
                        let ds = d - sdl;
                        if ds >= 0 { la[jp_v][dp] = lalpha[m][j as usize][ds as usize]; }
                    }
                }
            }
            if do_r_v && cp9b.rvalid[m] {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let js = j - sdr;
                    for dp in 0..ra[jp_v].len() {
                        let d = hdmin[v][jp_v] + dp as i32;
                        let ds = d - sdr;
                        if js >= 0 && ds >= 0 { ra[jp_v][dp] = ralpha[m][js as usize][ds as usize]; }
                    }
                }
            }
        }

        if stt == IL_ST || stt == ML_ST {
            // C 5163-5299: for j { for d { for y }}; J/L in one loop, R separate.
            if !state_is_detached(cm, v) && (do_j_v || do_l_v || do_r_v) {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    // yvalid for J/L mode (j valid for y).
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yctr in 0..cnum {
                        let yoffset = (yctr + nins_v) % cnum;
                        let y = cfirst + yoffset;
                        if j >= jmin[y] && j <= jmax[y] { yvalid.push(yoffset); }
                    }
                    if do_j_v || do_l_v {
                        let mut i = j - hdmin[v][jp_v] + 1;
                        for dp_v in 0..ja[jp_v].len() {
                            let d = hdmin[v][jp_v] + dp_v as i32;
                            let iu = i as usize;
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let jp_y = (j - jmin[y]) as usize;
                                let do_j_y = cp9b.jvalid[y];
                                let do_l_y = cp9b.lvalid[y] && fill_l;
                                if do_j_y || do_l_y {
                                    let dsm = d - sd;
                                    if dsm >= hdmin[y][jp_y] && dsm <= hdmax[y][jp_y] {
                                        let dpy = (dsm - hdmin[y][jp_y]) as usize;
                                        if do_j_v && do_j_y {
                                            let sc = if y == v { ja[jp_y][dpy] } else { jalpha[y][jp_y][dpy] };
                                            if sc > ja[jp_v][dp_v] { ja[jp_v][dp_v] = sc; jy[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                        }
                                        if do_l_v && do_l_y {
                                            let sc = if y == v { la[jp_y][dpy] } else { lalpha[y][jp_y][dpy] };
                                            if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; ly[jp_v][dp_v] = yoffset as i32 + TRMODE_L_OFFSET; }
                                        }
                                    }
                                }
                            }
                            // emission PP (raw i)
                            if do_j_v {
                                let jl = emit_mx.jl_pp[v].as_ref().unwrap();
                                ja[jp_v][dp_v] = flogsum(ja[jp_v][dp_v], jl[iu]);
                                if ja[jp_v][dp_v] < IMPOSSIBLE { ja[jp_v][dp_v] = IMPOSSIBLE; }
                                if !have_el && jy[jp_v][dp_v] == USED_EL && d > sd { ja[jp_v][dp_v] = IMPOSSIBLE; }
                            }
                            if do_l_v {
                                let ll = emit_mx.ll_pp[v].as_ref().unwrap();
                                if d >= 2 {
                                    la[jp_v][dp_v] = flogsum(la[jp_v][dp_v], ll[iu]);
                                    if !have_el && ly[jp_v][dp_v] == USED_EL { la[jp_v][dp_v] = IMPOSSIBLE; }
                                } else {
                                    la[jp_v][dp_v] = ll[iu];
                                    ly[jp_v][dp_v] = USED_TRUNC_END;
                                }
                                if la[jp_v][dp_v] < IMPOSSIBLE { la[jp_v][dp_v] = IMPOSSIBLE; }
                            }
                            i -= 1;
                        }
                    }
                    // handle R separately (uses 'd' not 'd-sd', disallow self-transit).
                    if do_r_v {
                        for dp_v in 0..ra[jp_v].len() {
                            let d = hdmin[v][jp_v] + dp_v as i32;
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                if y == v { continue; }
                                let jp_y = (j - jmin[y]) as usize;
                                let do_j_y = cp9b.jvalid[y];
                                let do_r_y = cp9b.rvalid[y] && fill_r;
                                if (do_j_y || do_r_y) && d >= hdmin[y][jp_y] && d <= hdmax[y][jp_y] {
                                    let dpy = (d - hdmin[y][jp_y]) as usize;
                                    if do_j_y {
                                        let sc = jalpha[y][jp_y][dpy];
                                        if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; ry[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                    }
                                    if do_r_y {
                                        let sc = ralpha[y][jp_y][dpy];
                                        if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; ry[jp_v][dp_v] = yoffset as i32 + TRMODE_R_OFFSET; }
                                    }
                                }
                            }
                            if ra[jp_v][dp_v] < IMPOSSIBLE { ra[jp_v][dp_v] = IMPOSSIBLE; }
                        }
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            // C 5300-5445: for j { for d { for y }}; J/R in one loop, L separate.
            if !state_is_detached(cm, v) && (do_j_v || do_l_v || do_r_v) {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    // yvalid for J/R mode (j-sdr valid for y).
                    let mut yvalid: Vec<usize> = Vec::new();
                    for yctr in 0..cnum {
                        let yoffset = (yctr + nins_v) % cnum;
                        let y = cfirst + yoffset;
                        if (j - sdr) >= jmin[y] && (j - sdr) <= jmax[y] { yvalid.push(yoffset); }
                    }
                    if do_j_v || do_r_v {
                        for dp_v in 0..ja[jp_v].len() {
                            let d = hdmin[v][jp_v] + dp_v as i32;
                            for &yoffset in &yvalid {
                                let y = cfirst + yoffset;
                                let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                                let do_j_y = cp9b.jvalid[y];
                                let do_r_y = cp9b.rvalid[y] && fill_r;
                                if do_j_y || do_r_y {
                                    let dsm = d - sd;
                                    if dsm >= hdmin[y][jp_y_sdr] && dsm <= hdmax[y][jp_y_sdr] {
                                        let dpy = (dsm - hdmin[y][jp_y_sdr]) as usize;
                                        if do_j_v && do_j_y {
                                            let sc = if y == v { ja[jp_y_sdr][dpy] } else { jalpha[y][jp_y_sdr][dpy] };
                                            if sc > ja[jp_v][dp_v] { ja[jp_v][dp_v] = sc; jy[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                        }
                                        if do_r_v && do_r_y {
                                            let sc = if y == v { ra[jp_y_sdr][dpy] } else { ralpha[y][jp_y_sdr][dpy] };
                                            if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; ry[jp_v][dp_v] = yoffset as i32 + TRMODE_R_OFFSET; }
                                        }
                                    }
                                }
                            }
                            // emission PP (raw j)
                            if do_j_v {
                                let jr = emit_mx.jr_pp[v].as_ref().unwrap();
                                ja[jp_v][dp_v] = flogsum(ja[jp_v][dp_v], jr[j as usize]);
                                if ja[jp_v][dp_v] < IMPOSSIBLE { ja[jp_v][dp_v] = IMPOSSIBLE; }
                                if !have_el && jy[jp_v][dp_v] == USED_EL && d > sd { ja[jp_v][dp_v] = IMPOSSIBLE; }
                            }
                            if do_r_v {
                                let rr = emit_mx.rr_pp[v].as_ref().unwrap();
                                if d >= 2 {
                                    ra[jp_v][dp_v] = flogsum(ra[jp_v][dp_v], rr[j as usize]);
                                    if !have_el && ry[jp_v][dp_v] == USED_EL { ra[jp_v][dp_v] = IMPOSSIBLE; }
                                } else {
                                    ra[jp_v][dp_v] = rr[j as usize];
                                    ry[jp_v][dp_v] = USED_TRUNC_END;
                                }
                                if ra[jp_v][dp_v] < IMPOSSIBLE { ra[jp_v][dp_v] = IMPOSSIBLE; }
                            }
                        }
                    }
                    // handle L separately (uses 'j' and 'd', disallow self-transit).
                    if do_l_v {
                        let mut yvalid_l: Vec<usize> = Vec::new();
                        for yctr in 0..cnum {
                            let yoffset = (yctr + nins_v) % cnum;
                            let y = cfirst + yoffset;
                            if j >= jmin[y] && j <= jmax[y] { yvalid_l.push(yoffset); }
                        }
                        for dp_v in 0..la[jp_v].len() {
                            let d = hdmin[v][jp_v] + dp_v as i32;
                            for &yoffset in &yvalid_l {
                                let y = cfirst + yoffset;
                                if y == v { continue; }
                                let jp_y = (j - jmin[y]) as usize;
                                let do_j_y = cp9b.jvalid[y];
                                let do_l_y = cp9b.lvalid[y] && fill_l;
                                if (do_j_y || do_l_y) && d >= hdmin[y][jp_y] && d <= hdmax[y][jp_y] {
                                    let dpy = (d - hdmin[y][jp_y]) as usize;
                                    if do_j_y {
                                        let sc = jalpha[y][jp_y][dpy];
                                        if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; ly[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                    }
                                    if do_l_y {
                                        let sc = lalpha[y][jp_y][dpy];
                                        if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; ly[jp_v][dp_v] = yoffset as i32 + TRMODE_L_OFFSET; }
                                    }
                                }
                            }
                            if la[jp_v][dp_v] < IMPOSSIBLE { la[jp_v][dp_v] = IMPOSSIBLE; }
                        }
                    }
                }
            }
        } else if stt == MP_ST {
            // C 5446-5723: for y { J/L/R band-clamp } then emission then clamp then !have_el.
            if do_j_v || do_l_v || do_r_v {
                for yctr in 0..cnum {
                    let yoffset = (yctr + nins_v) % cnum;
                    let y = cfirst + yoffset;
                    let do_j_y = cp9b.jvalid[y];
                    let do_l_y = cp9b.lvalid[y] && fill_l;
                    let do_r_y = cp9b.rvalid[y] && fill_r;
                    if do_j_v && do_j_y {
                        let jn = jmin[v].max(jmin[y] + sdr);
                        let jx = jmax[v].min(jmax[y] + sdr);
                        let mut jp_y_sdr = jn - jmin[y] - sdr;
                        for jp_v in (jn - jmin[v])..=(jx - jmin[v]) {
                            let jp_v = jp_v as usize;
                            let jys = jp_y_sdr as usize;
                            let dn = hdmin[v][jp_v].max(hdmin[y][jys] + sd);
                            let dx = hdmax[v][jp_v].min(hdmax[y][jys] + sd);
                            let mut dp_y_sd = dn - hdmin[y][jys] - sd;
                            for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                let dp_v = dp_v as usize;
                                let sc = jalpha[y][jys][dp_y_sd as usize];
                                if sc > ja[jp_v][dp_v] { ja[jp_v][dp_v] = sc; jy[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                dp_y_sd += 1;
                            }
                            jp_y_sdr += 1;
                        }
                    }
                    if do_l_v && (do_j_y || do_l_y) {
                        let jn = jmin[v].max(jmin[y]);
                        let jx = jmax[v].min(jmax[y]);
                        let mut jp_y = jn - jmin[y];
                        for jp_v in (jn - jmin[v])..=(jx - jmin[v]) {
                            let jp_v = jp_v as usize;
                            let jyi = jp_y as usize;
                            let dn = hdmin[v][jp_v].max(hdmin[y][jyi] + sdl);
                            let dx = hdmax[v][jp_v].min(hdmax[y][jyi] + sdl);
                            if do_j_y {
                                let mut dp_y_sdl = dn - hdmin[y][jyi] - sdl;
                                for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                    let dp_v = dp_v as usize;
                                    let sc = jalpha[y][jyi][dp_y_sdl as usize];
                                    if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; ly[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                    dp_y_sdl += 1;
                                }
                            }
                            if do_l_y {
                                let mut dp_y_sdl = dn - hdmin[y][jyi] - sdl;
                                for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                    let dp_v = dp_v as usize;
                                    let sc = lalpha[y][jyi][dp_y_sdl as usize];
                                    if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; ly[jp_v][dp_v] = yoffset as i32 + TRMODE_L_OFFSET; }
                                    dp_y_sdl += 1;
                                }
                            }
                            jp_y += 1;
                        }
                    }
                    if do_r_v && (do_j_y || do_r_y) {
                        let jn = jmin[v].max(jmin[y] + sdr);
                        let jx = jmax[v].min(jmax[y] + sdr);
                        let mut jp_y_sdr = jn - jmin[y] - sdr;
                        for jp_v in (jn - jmin[v])..=(jx - jmin[v]) {
                            let jp_v = jp_v as usize;
                            let jys = jp_y_sdr as usize;
                            let dn = hdmin[v][jp_v].max(hdmin[y][jys] + sdr);
                            let dx = hdmax[v][jp_v].min(hdmax[y][jys] + sdr);
                            if do_j_y {
                                let mut dp_y_sdr = dn - hdmin[y][jys] - sdr;
                                for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                    let dp_v = dp_v as usize;
                                    let sc = jalpha[y][jys][dp_y_sdr as usize];
                                    if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; ry[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                    dp_y_sdr += 1;
                                }
                            }
                            if do_r_y {
                                let mut dp_y_sdr = dn - hdmin[y][jys] - sdr;
                                for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                    let dp_v = dp_v as usize;
                                    let sc = ralpha[y][jys][dp_y_sdr as usize];
                                    if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; ry[jp_v][dp_v] = yoffset as i32 + TRMODE_R_OFFSET; }
                                    dp_y_sdr += 1;
                                }
                            }
                            jp_y_sdr += 1;
                        }
                    }
                }
            }
            // emission (C 5570-5620), raw i and raw j.
            if do_j_v {
                let jl = emit_mx.jl_pp[v].as_ref().unwrap();
                let jr = emit_mx.jr_pp[v].as_ref().unwrap();
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin[v][jp_v] + 1;
                    for dp_v in 0..ja[jp_v].len() {
                        let d = hdmin[v][jp_v] + dp_v as i32;
                        if d >= 2 {
                            ja[jp_v][dp_v] = flogsum(ja[jp_v][dp_v], flogsum(jl[i as usize], jr[j as usize]));
                        } else {
                            ja[jp_v][dp_v] = IMPOSSIBLE;
                        }
                        i -= 1;
                    }
                }
            }
            if do_l_v {
                let ll = emit_mx.ll_pp[v].as_ref().unwrap();
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin[v][jp_v] + 1;
                    for dp_v in 0..la[jp_v].len() {
                        let d = hdmin[v][jp_v] + dp_v as i32;
                        if d >= 2 {
                            la[jp_v][dp_v] = flogsum(la[jp_v][dp_v], ll[i as usize]);
                        } else {
                            la[jp_v][dp_v] = ll[i as usize];
                            ly[jp_v][dp_v] = USED_TRUNC_END;
                        }
                        i -= 1;
                    }
                }
            }
            if do_r_v {
                let rr = emit_mx.rr_pp[v].as_ref().unwrap();
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    for dp_v in 0..ra[jp_v].len() {
                        let d = hdmin[v][jp_v] + dp_v as i32;
                        if d >= 2 {
                            ra[jp_v][dp_v] = flogsum(ra[jp_v][dp_v], rr[j as usize]);
                        } else {
                            ra[jp_v][dp_v] = rr[j as usize];
                            ry[jp_v][dp_v] = USED_TRUNC_END;
                        }
                    }
                }
            }
            // clamp + !have_el EL-emission disallow (C 5625-5723).
            if do_j_v {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    for dp_v in 0..ja[jp_v].len() { if ja[jp_v][dp_v] < IMPOSSIBLE { ja[jp_v][dp_v] = IMPOSSIBLE; } }
                    if !have_el {
                        for dp_v in 0..ja[jp_v].len() {
                            let d = hdmin[v][jp_v] + dp_v as i32;
                            if jy[jp_v][dp_v] == USED_EL && d > sd { ja[jp_v][dp_v] = IMPOSSIBLE; }
                        }
                    }
                }
            }
            if do_l_v {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    for dp_v in 0..la[jp_v].len() { if la[jp_v][dp_v] < IMPOSSIBLE { la[jp_v][dp_v] = IMPOSSIBLE; } }
                    if !have_el {
                        for dp_v in 0..la[jp_v].len() {
                            let d = hdmin[v][jp_v] + dp_v as i32;
                            if ly[jp_v][dp_v] == USED_EL && d > sdl { la[jp_v][dp_v] = IMPOSSIBLE; }
                        }
                    }
                }
            }
            if do_r_v {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    for dp_v in 0..ra[jp_v].len() { if ra[jp_v][dp_v] < IMPOSSIBLE { ra[jp_v][dp_v] = IMPOSSIBLE; } }
                    if !have_el {
                        for dp_v in 0..ra[jp_v].len() {
                            let d = hdmin[v][jp_v] + dp_v as i32;
                            if ry[jp_v][dp_v] == USED_EL && d > sdr { ra[jp_v][dp_v] = IMPOSSIBLE; }
                        }
                    }
                }
            }
        } else if !is_b {
            // D, S states (C 5725-5806): for y { J/L/R band-clamp } + d==0 specials.
            if do_j_v || do_l_v || do_r_v {
                for yctr in 0..cnum {
                    let yoffset = (yctr + nins_v) % cnum;
                    let y = cfirst + yoffset;
                    let do_j_y = cp9b.jvalid[y];
                    let do_l_y = cp9b.lvalid[y] && fill_l;
                    let do_r_y = cp9b.rvalid[y] && fill_r;
                    if do_j_v && do_j_y {
                        let jn = jmin[v].max(jmin[y]);
                        let jx = jmax[v].min(jmax[y]);
                        let mut jp_y = jn - jmin[y];
                        for jp_v in (jn - jmin[v])..=(jx - jmin[v]) {
                            let jp_v = jp_v as usize;
                            let jyi = jp_y as usize;
                            let dn = hdmin[v][jp_v].max(hdmin[y][jyi]);
                            let dx = hdmax[v][jp_v].min(hdmax[y][jyi]);
                            let mut dp_y = dn - hdmin[y][jyi];
                            for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                let dp_v = dp_v as usize;
                                let sc = jalpha[y][jyi][dp_y as usize];
                                if sc > ja[jp_v][dp_v] { ja[jp_v][dp_v] = sc; jy[jp_v][dp_v] = yoffset as i32 + TRMODE_J_OFFSET; }
                                dp_y += 1;
                            }
                            jp_y += 1;
                        }
                    }
                    if do_l_v && do_l_y {
                        let jn = jmin[v].max(jmin[y]);
                        let jx = jmax[v].min(jmax[y]);
                        let mut jp_y = jn - jmin[y];
                        for jp_v in (jn - jmin[v])..=(jx - jmin[v]) {
                            let jp_v = jp_v as usize;
                            let jyi = jp_y as usize;
                            let dn = hdmin[v][jp_v].max(hdmin[y][jyi]);
                            let dx = hdmax[v][jp_v].min(hdmax[y][jyi]);
                            let mut dp_y = dn - hdmin[y][jyi];
                            for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                let dp_v = dp_v as usize;
                                let sc = lalpha[y][jyi][dp_y as usize];
                                if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; ly[jp_v][dp_v] = yoffset as i32 + TRMODE_L_OFFSET; }
                                dp_y += 1;
                            }
                            jp_y += 1;
                        }
                    }
                    if do_r_v && do_r_y {
                        let jn = jmin[v].max(jmin[y]);
                        let jx = jmax[v].min(jmax[y]);
                        let mut jp_y = jn - jmin[y];
                        for jp_v in (jn - jmin[v])..=(jx - jmin[v]) {
                            let jp_v = jp_v as usize;
                            let jyi = jp_y as usize;
                            let dn = hdmin[v][jp_v].max(hdmin[y][jyi]);
                            let dx = hdmax[v][jp_v].min(hdmax[y][jyi]);
                            let mut dp_y = dn - hdmin[y][jyi];
                            for dp_v in (dn - hdmin[v][jp_v])..=(dx - hdmin[v][jp_v]) {
                                let dp_v = dp_v as usize;
                                let sc = ralpha[y][jyi][dp_y as usize];
                                if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; ry[jp_v][dp_v] = yoffset as i32 + TRMODE_R_OFFSET; }
                                dp_y += 1;
                            }
                            jp_y += 1;
                        }
                    }
                }
            }
            // d==0 specials (C 5757-5806): force L/R IMPOSSIBLE at hdmin==0; S gets USED_TRUNC_END.
            if do_l_v {
                for jp_v in 0..njv(v) {
                    if hdmin[v][jp_v] == 0 {
                        la[jp_v][0] = IMPOSSIBLE;
                        if stt == S_ST { ly[jp_v][0] = USED_TRUNC_END; }
                    }
                }
            }
            if do_r_v {
                for jp_v in 0..njv(v) {
                    if hdmin[v][jp_v] == 0 {
                        ra[jp_v][0] = IMPOSSIBLE;
                        if stt == S_ST { ry[jp_v][0] = USED_TRUNC_END; }
                    }
                }
            }
        } else {
            // B_st (C 5808-6008): main k-loop (6 inequalities) + 2 special L/R cases.
            let y = cfirst; // BEGL_S (left)
            let z = cnum; // BEGR_S (right)
            let do_j_y = cp9b.jvalid[y];
            let do_l_y = cp9b.lvalid[y] && fill_l;
            let do_r_y = cp9b.rvalid[y] && fill_r;
            let do_j_z = cp9b.jvalid[z];
            let do_l_z = cp9b.lvalid[z] && fill_l;
            let do_r_z = cp9b.rvalid[z] && fill_r;
            if do_j_v || do_l_v || do_r_v || do_t_v {
                let jn = jmin[v].max(jmin[z]);
                let jx = jmax[v].min(jmax[z]);
                for j in jn..=jx {
                    let jp_v = (j - jmin[v]) as usize;
                    let jp_y = j - jmin[y];
                    let jp_z = (j - jmin[z]) as usize;
                    let mut kn = (j - jmax[y]).max(hdmin[z][jp_z]);
                    kn = kn.max(0);
                    let kx = jp_y.min(hdmax[z][jp_z]);
                    for dp_v in 0..ja[jp_v].len() {
                        let d = hdmin[v][jp_v] + dp_v as i32;
                        let mut k = kn;
                        while k <= kx {
                            let jpy_k = (jp_y - k) as usize; // = (j-k)-jmin[y] >= 0
                            if k >= d - hdmax[y][jpy_k] && k <= d - hdmin[y][jpy_k] {
                                let kp_z = (k - hdmin[z][jp_z]) as usize;
                                let dp_y = d - hdmin[y][jpy_k]; // >= k
                                let dyk = (dp_y - k) as usize;
                                if do_j_v && do_j_y && do_j_z {
                                    let left = jalpha[y][jpy_k][dyk];
                                    let right = jalpha[z][jp_z][kp_z];
                                    if (not_impossible(left) || d == k) && (not_impossible(right) || k == 0) {
                                        let sc = flogsum(left, right);
                                        if sc > ja[jp_v][dp_v] { ja[jp_v][dp_v] = sc; jk[jp_v][dp_v] = k; }
                                    }
                                }
                                if do_l_v && do_j_y && do_l_z {
                                    let left = jalpha[y][jpy_k][dyk];
                                    let lz = lalpha[z][jp_z][kp_z];
                                    if (not_impossible(left) || d == k) && (not_impossible(lz) || k == 0) {
                                        let sc = flogsum(left, lz);
                                        if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; lk[jp_v][dp_v] = k; }
                                    }
                                }
                                if do_r_v && do_r_y && do_j_z {
                                    let left = ralpha[y][jpy_k][dyk];
                                    let right = jalpha[z][jp_z][kp_z];
                                    if (not_impossible(left) || d == k) && (not_impossible(right) || k == 0) {
                                        let sc = flogsum(left, right);
                                        if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; rk[jp_v][dp_v] = k; }
                                    }
                                }
                                if k != 0 && k != d && do_t_v && do_r_y && do_l_z {
                                    let left = ralpha[y][jpy_k][dyk];
                                    let right = lalpha[z][jp_z][kp_z];
                                    if not_impossible(left) && not_impossible(right) {
                                        let sc = flogsum(left, right);
                                        if sc > ta[jp_v][dp_v] { ta[jp_v][dp_v] = sc; tk[jp_v][dp_v] = k; }
                                    }
                                }
                            }
                            k += 1;
                        }
                    }
                }
                // special case 1: k==0, full seq on left (C 5924-5966).
                if do_l_v && (do_j_y || do_l_y) {
                    let jn = jmin[v].max(jmin[y]);
                    let jx = jmax[v].min(jmax[y]);
                    for j in jn..=jx {
                        let jp_v = (j - jmin[v]) as usize;
                        let jp_y = (j - jmin[y]) as usize;
                        let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                        let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                        for d in dn..=dx {
                            let dp_v = (d - hdmin[v][jp_v]) as usize;
                            let dp_y = (d - hdmin[y][jp_y]) as usize;
                            if do_j_y {
                                let sc = jalpha[y][jp_y][dp_y];
                                if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; lk[jp_v][dp_v] = 0; lkm[jp_v][dp_v] = TRMODE_J; }
                            }
                            if do_l_y {
                                let sc = lalpha[y][jp_y][dp_y];
                                if sc > la[jp_v][dp_v] { la[jp_v][dp_v] = sc; lk[jp_v][dp_v] = 0; lkm[jp_v][dp_v] = TRMODE_L; }
                            }
                        }
                    }
                }
                // special case 2: k==d, full seq on right (C 5967-6008).
                if do_r_v && (do_j_z || do_r_z) {
                    let jn = jmin[v].max(jmin[z]);
                    let jx = jmax[v].min(jmax[z]);
                    for j in jn..=jx {
                        let jp_v = (j - jmin[v]) as usize;
                        let jp_z = (j - jmin[z]) as usize;
                        let dn = hdmin[v][jp_v].max(hdmin[z][jp_z]);
                        let dx = hdmax[v][jp_v].min(hdmax[z][jp_z]);
                        for d in dn..=dx {
                            let dp_v = (d - hdmin[v][jp_v]) as usize;
                            let dp_z = (d - hdmin[z][jp_z]) as usize;
                            if do_j_z {
                                let sc = jalpha[z][jp_z][dp_z];
                                if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; rk[jp_v][dp_v] = d; rkm[jp_v][dp_v] = TRMODE_J; }
                            }
                            if do_r_z {
                                let sc = ralpha[z][jp_z][dp_z];
                                if sc > ra[jp_v][dp_v] { ra[jp_v][dp_v] = sc; rk[jp_v][dp_v] = d; rkm[jp_v][dp_v] = TRMODE_R; }
                            }
                        }
                    }
                }
            }
        }

        // ROOT truncated-begin update (C 6002-6032): single best b for preset_mode.
        if not_impossible(pty[v]) && l >= jmin[v] && l <= jmax[v] {
            let jp_vr = (l - jmin[v]) as usize;
            if l >= hdmin[v][jp_vr] && l <= hdmax[v][jp_vr] {
                let lpr = (l - hdmin[v][jp_vr]) as usize;
                match preset_mode {
                    TRMODE_J => { if do_j_v { let c = ja[jp_vr][lpr]; if c > best { best = c; b = v as i32; } } }
                    TRMODE_L => { if do_l_v { let c = la[jp_vr][lpr]; if c > best { best = c; b = v as i32; } } }
                    TRMODE_R => { if do_r_v { let c = ra[jp_vr][lpr]; if c > best { best = c; b = v as i32; } } }
                    _ => { if do_t_v && is_b { let c = ta[jp_vr][lpr]; if c > best { best = c; b = v as i32; } } }
                }
            }
        }

        jalpha[v] = ja;
        if fill_l { lalpha[v] = la; }
        if fill_r { ralpha[v] = ra; }
        jysh[v] = jy;
        if fill_l { lysh[v] = ly; }
        if fill_r { rysh[v] = ry; }
        if is_b {
            jksh[v] = jk;
            if fill_l { lksh[v] = lk; lkmode[v] = lkm; }
            if fill_r { rksh[v] = rk; rkmode[v] = rkm; }
            if fill_t { tksh[v] = tk; talpha[v] = ta; }
        }
    }

    let pp = (sre_exp2(best) / (w as f32) as f64) as f32;

    // ---- traceback: OptAcc branch of cm_tr_alignT_hb (cm_dpalign_trunc.c:471) ----
    let mut mode = preset_mode;
    let mut tr = Parsetree::new(w + 4);
    tr.is_std = false; // C cm_tr_alignT_hb (optacc): truncated parse
    tr.pass_idx = pass_idx;
    // C cm_dpalign_trunc.c:540: tr->trpenalty = (local?l:g)_ptyAA[pty_idx][b].
    tr.trpenalty = pty[b as usize];
    tr.add_node_mode(1, l, 0, -1, -1, -1, mode); // attach root ROOT_S
    let mut pda_i: Vec<i32> = Vec::new();
    let mut pda_c: Vec<i8> = Vec::new();
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = l;
    let mut d: i32 = l;

    loop {
        let vu = v as usize;
        // super-special: BEGL_S/BEGR_S d==0, mode/band-disallowed (C 553-593).
        let mut allow_s_trunc_end = false;
        let mut allow_s_local_end = false;
        let mut jp_v = 0usize;
        let mut dp_v = 0usize;
        if v != cm.m && cm.sttype[vu] as i32 != E_ST {
            let stid = cm.stid[vu] as i32;
            let mode_disallowed = (mode == TRMODE_J && !cp9b.jvalid[vu])
                || (mode == TRMODE_L && !cp9b.lvalid[vu])
                || (mode == TRMODE_R && !cp9b.rvalid[vu]);
            let out_of_band = j < jmin[vu]
                || j > jmax[vu]
                || {
                    let jpv = j - jmin[vu];
                    jpv < 0 || d < hdmin[vu][jpv as usize] || d > hdmax[vu][jpv as usize]
                };
            if (stid == BEGL_S || stid == BEGR_S) && d == 0 && (mode_disallowed || out_of_band) {
                if (stid == BEGL_S && mode == TRMODE_R) || (stid == BEGR_S && mode == TRMODE_L) {
                    allow_s_trunc_end = true;
                } else {
                    // do_optacc == TRUE: allow a d==0 local end out of this start state.
                    allow_s_local_end = true;
                }
            } else if cm.sttype[vu] as i32 != EL_ST {
                jp_v = (j - jmin[vu]) as usize;
                dp_v = (d - hdmin[vu][jp_v]) as usize;
            }
        }

        if v != cm.m && cm.sttype[vu] as i32 == B_ST {
            let k = match mode {
                TRMODE_J => jksh[vu][jp_v][dp_v],
                TRMODE_L => lksh[vu][jp_v][dp_v],
                TRMODE_R => rksh[vu][jp_v][dp_v],
                _ => tksh[vu][jp_v][dp_v],
            };
            let prvmode = mode;
            let rmode = match mode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => TRMODE_L,
                TRMODE_R => rkmode[vu][jp_v][dp_v],
                _ => TRMODE_L,
            };
            let bpar = tr.n - 1;
            pda_c.push(rmode);
            pda_i.push(j);
            pda_i.push(k);
            pda_i.push(bpar);
            let lmode = match prvmode {
                TRMODE_J => TRMODE_J,
                TRMODE_L => lkmode[vu][jp_v][dp_v],
                TRMODE_R => TRMODE_R,
                _ => TRMODE_R,
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
        } else if v == cm.m || cm.sttype[vu] as i32 == E_ST || cm.sttype[vu] as i32 == EL_ST {
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
        } else {
            let yoffset_raw = if allow_s_trunc_end {
                USED_TRUNC_END
            } else if allow_s_local_end {
                USED_EL
            } else if v == 0 {
                USED_TRUNC_BEGIN
            } else {
                match mode {
                    TRMODE_J => jysh[vu][jp_v][dp_v],
                    TRMODE_L => lysh[vu][jp_v][dp_v],
                    TRMODE_R => rysh[vu][jp_v][dp_v],
                    _ => USED_TRUNC_BEGIN,
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

            let stt = cm.sttype[vu] as i32;
            match stt {
                x if x == MP_ST => {
                    if mode == TRMODE_J { i += 1; }
                    if mode == TRMODE_L && d > 0 { i += 1; }
                    if mode == TRMODE_J { j -= 1; }
                    if mode == TRMODE_R && d > 0 { j -= 1; }
                }
                x if x == ML_ST || x == IL_ST => {
                    if mode == TRMODE_J { i += 1; }
                    if mode == TRMODE_L && d > 0 { i += 1; }
                }
                x if x == MR_ST || x == IR_ST => {
                    if mode == TRMODE_J { j -= 1; }
                    if mode == TRMODE_R && d > 0 { j -= 1; }
                }
                _ => {}
            }
            d = j - i + 1;

            if yoffset == USED_EL || yoffset == USED_TRUNC_END {
                if yoffset == USED_EL {
                    let idx = tr.add_node_mode(i, j, cm.m, -1, -1, tr.n - 1, mode);
                    tr.nxtl[(tr.n - 2) as usize] = idx;
                }
                v = cm.m;
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
    }

    (tr, pp)
}

/// C `cm_TrPostCode` (cm_dpalign_trunc.c:9619): build the PP annotation string by
/// walking the parsetree, indexing the per-mode emit posteriors. Returns (ppstr, avgpp).
pub fn tr_postcode(cm: &CM, lp: i32, emit_mx: &TrEmitMx, tr: &Parsetree) -> (Vec<u8>, f32) {
    let l = lp as usize;
    let mut ppstr = vec![0u8; l];
    let mut sum_logp = IMPOSSIBLE;
    for x in 0..tr.n as usize {
        let v = tr.state[x] as usize;
        let i = tr.emitl[x];
        let j = tr.emitr[x];
        let mode = tr.mode[x];
        let stt = if v == cm.m as usize { EL_ST } else { cm.sttype[v] as i32 };
        if stt == EL_ST {
            if mode == TRMODE_J || mode == TRMODE_L || mode == TRMODE_R {
                for r in i..=j {
                    let cur = match mode {
                        TRMODE_J => emit_mx.jl_pp[v].as_ref().unwrap()[r as usize],
                        TRMODE_L => emit_mx.ll_pp[v].as_ref().unwrap()[r as usize],
                        _ => emit_mx.rr_pp[v].as_ref().unwrap()[r as usize],
                    };
                    ppstr[(r - 1) as usize] = fscore2postcode(cur);
                    sum_logp = flogsum(sum_logp, cur);
                }
            }
        }
        if stt == MP_ST || stt == ML_ST || stt == IL_ST {
            if mode == TRMODE_J || mode == TRMODE_L {
                let cur = if mode == TRMODE_J {
                    emit_mx.jl_pp[v].as_ref().unwrap()[i as usize]
                } else {
                    emit_mx.ll_pp[v].as_ref().unwrap()[i as usize]
                };
                ppstr[(i - 1) as usize] = fscore2postcode(cur);
                sum_logp = flogsum(sum_logp, cur);
            }
        }
        if stt == MP_ST || stt == MR_ST || stt == IR_ST {
            if mode == TRMODE_J || mode == TRMODE_R {
                let cur = if mode == TRMODE_J {
                    emit_mx.jr_pp[v].as_ref().unwrap()[j as usize]
                } else {
                    emit_mx.rr_pp[v].as_ref().unwrap()[j as usize]
                };
                ppstr[(j - 1) as usize] = fscore2postcode(cur);
                sum_logp = flogsum(sum_logp, cur);
            }
        }
    }
    let avgp = (sre_exp2(sum_logp) / l as f64) as f32;
    (ppstr, avgp)
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
        // C ParsetreeToCMBounds (cm_parsetree.c:2655-2672): a local end (v==cm->M) is
        // treated as BOTH left- and right-emitting, using the node it "replaced"
        // (nd = 1 + ndidx[prev state]); it contributes to cfrom_emit/cto_emit
        // (forced, regardless of mode) but not to first/final_emit (sdl=sdr=0). Skipping
        // it was a bug: a parse that ELs over a consensus stretch (e.g. a tiny 3' hit
        // that local-begins deep and ELs the 5' interior) then reported cfrom_emit at
        // the emitted region only, leaving cfrom_emit != cfrom_span → a spurious 5'
        // truncation marker in the STD-pass alidisplay.
        let (nd, lpos, rpos, is_left, is_right, insert_sd, mode, force_el) = if v == cm.m {
            let prv_v = tr.state[ti - 1] as usize;
            let nd = 1 + cm.ndidx[prv_v] as usize; // node the EL replaced
            (nd, node_lpos(nd), node_rpos(nd), true, true, 0, tr.mode[ti - 1], true)
        } else {
            let vu = v as usize;
            let stt = cm.sttype[vu] as i32;
            let nd = cm.ndidx[vu] as usize;
            let ndt = cm.ndtype[nd] as i32;
            let (il, ir, isd) = if stt == IL_ST {
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
            (nd, node_lpos(nd), node_rpos(nd), il, ir, isd, tr.mode[ti], false)
        };
        let _ = nd;
        let emits_left = force_el || mode == TRMODE_J || mode == TRMODE_L;
        let emits_right = force_el || mode == TRMODE_J || mode == TRMODE_R;
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
    // C cm_parsetree.c:2708-2711: the internal ANY pass spans the full model
    // unconditionally (no have_i0/have_j0 gate). The truncation type then follows
    // purely from cfrom_emit>1 (5' trunc) / cto_emit<clen (3' trunc) in the emitted
    // parse, so an internal hit renders "5'&3'" (both spans truncated).
    if pass_idx == PLI_PASS_5P_AND_3P_ANY {
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
pub fn marginal_emissions(cm: &CM) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // Faithful: the full per-state augmented-alphabet marginal arrays filled by
    // cm_logoddsify (STEP 0). `cm` must have been logoddsified.
    (cm.lmesc.clone(), cm.rmesc.clone())
}

// ===========================================================================
// cm_TrStochasticParsetreeHB (cm_parsetree.c:3813) — sample a parsetree from a
// HMM-banded truncated float Inside matrix (filled by tr_inside_align_hb). Used
// by cmbuild --refine --gibbs (default truncated+HB config). RNG draw order must
// match C exactly: shares sample_helper (cm_dpalign.rs) which consumes esl_random
// via esl_rnd_FChoose. Faithful transcription of the C traceback including all
// band offset arithmetic. GLOBAL or LOCAL config (endsc/EL handled per flags).
// ===========================================================================

// C cm.c:ModeEmitsLeft / ModeEmitsRight.
#[inline]
fn mode_emits_left(mode: i8) -> bool {
    mode == TRMODE_J || mode == TRMODE_L
}
#[inline]
fn mode_emits_right(mode: i8) -> bool {
    mode == TRMODE_J || mode == TRMODE_R
}

// C cm_parsetree.c:get_femission_score_trunc.
#[inline]
fn get_femission_score_trunc(
    cm: &CM,
    dsq: &[u8],
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    v: usize,
    i: i32,
    j: i32,
    mode: i8,
) -> f32 {
    let kp = crate::cm::ALPHABET_SIZE_P;
    let stt = cm.sttype[v] as i32;
    if stt == ML_ST || stt == IL_ST {
        if mode_emits_left(mode) {
            cm.oesc[v][dsq[i as usize] as usize]
        } else {
            0.0
        }
    } else if stt == MR_ST || stt == IR_ST {
        if mode_emits_right(mode) {
            cm.oesc[v][dsq[j as usize] as usize]
        } else {
            0.0
        }
    } else if stt == MP_ST {
        if mode_emits_left(mode) && mode_emits_right(mode) {
            cm.oesc[v][dsq[i as usize] as usize * kp + dsq[j as usize] as usize]
        } else if mode_emits_left(mode) {
            lm[v][dsq[i as usize] as usize]
        } else if mode_emits_right(mode) {
            rm[v][dsq[j as usize] as usize]
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// C `cm_TrStochasticParsetreeHB` (cm_parsetree.c:3813). Samples a parsetree from
/// the HMM-banded truncated Inside matrix `mx` (from [`tr_inside_align_hb`]).
/// `preset_mode` is the mode fixed by the Inside fill (TRMODE_J/L/R/T) or
/// [`TRMODE_UNKNOWN`] to sample the mode. `use_local` selects the local vs global
/// truncation-penalty array (C: `cm->flags & CMH_LOCAL_BEGIN`). Returns
/// `(parsetree, sampled_mode, fsc)`.
#[allow(clippy::too_many_arguments)]
pub fn tr_stochastic_parsetree_hb(
    cm: &CM,
    trp: &TrPenalties,
    cp9b: &crate::cp9::CP9Bands,
    mx: &TrHbMx,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    l: i32,
    pass_idx: i32,
    preset_mode: i8,
    use_local: bool,
    r: &mut crate::easel::random::EslRandom,
) -> (Parsetree, i8, f32) {
    let m = cm.m as usize;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let ja = &mx.j;
    let la = &mx.l;
    let ra = &mx.r;
    let ta = &mx.t;
    let maxconnect = crate::constants::MAXCONNECT as usize;
    let pty = trp.pty_slice(pass_idx, use_local);
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let local_end = cm.flags & CMH_LOCAL_END != 0;

    // filled_L/R/T from preset_mode (C cm_TrFillFromMode).
    let (filled_l, filled_r, filled_t) = match preset_mode {
        TRMODE_J => (false, false, false),
        TRMODE_L => (true, false, false),
        TRMODE_R => (false, true, false),
        _ => (true, true, true),
    };

    // Multipurpose vectors (C vec_size = max(L+3, max(M, 3*MAXCONNECT+1))).
    let vec_size = (l + 3).max((cm.m).max(3 * maxconnect as i32 + 1)) as usize;
    let mut pa = vec![IMPOSSIBLE; vec_size];
    let mut y_mode_a = vec![TRMODE_UNKNOWN; vec_size];
    let mut z_mode_a = vec![TRMODE_UNKNOWN; vec_size];
    let mut ka = vec![0i32; vec_size];
    let mut yoffset_a = vec![0i32; vec_size];
    // Per-mode transition vectors (MAXCONNECT+1).
    let mut jpa = vec![IMPOSSIBLE; maxconnect + 1];
    let mut lpa = vec![IMPOSSIBLE; maxconnect + 1];
    let mut rpa = vec![IMPOSSIBLE; maxconnect + 1];

    let jp_0 = (l - jmin[0]) as usize;
    let lp_0 = (l - hdmin[0][jp_0]) as usize;

    // Sample the marginal mode if unknown.
    let mut parsetree_mode = preset_mode;
    if preset_mode == TRMODE_UNKNOWN {
        for p in pa.iter_mut().take(4) {
            *p = IMPOSSIBLE;
        }
        if cp9b.jvalid[0] {
            pa[0] = ja[0][jp_0][lp_0];
        }
        if cp9b.lvalid[0] {
            pa[1] = la[0][jp_0][lp_0];
        }
        if cp9b.rvalid[0] {
            pa[2] = ra[0][jp_0][lp_0];
        }
        if cp9b.tvalid[0] {
            pa[3] = ta[0][jp_0][lp_0];
        }
        let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..4])
            .expect("tr_stochastic_parsetree_hb: no valid alignment modes");
        parsetree_mode = match choice {
            0 => TRMODE_J,
            1 => TRMODE_L,
            2 => TRMODE_R,
            _ => TRMODE_T,
        };
    }

    // Init parse tree with the root S state (mode = parsetree_mode).
    let mut tr = Parsetree::new(100);
    // C cm_TrStochasticParsetreeHB:3941: tr->is_std = FALSE (truncated parse).
    tr.is_std = false;
    tr.pass_idx = pass_idx;
    tr.add_node_mode(1, l, 0, -1, -1, -1, parsetree_mode);

    // Stacks for bifurcation right-fragment recovery.
    let mut i_pda: Vec<i32> = Vec::new(); // pushes j, k, parent-trace-index
    let mut c_pda: Vec<i8> = Vec::new(); // right-fragment mode

    let mut v: usize = 0;
    let mut j = l;
    let mut d = l;
    let mut i = 1;
    let mut v_mode = parsetree_mode;
    let mut fsc = 0.0f32;

    loop {
        let stt = cm.sttype[v] as i32;
        let stid = cm.stid[v] as i32;
        // Super-special truncated-end case (d==0, mode UNKNOWN, BEGL_S/BEGR_S).
        let mut allow_s_trunc_end = false;
        if d == 0 && v_mode == TRMODE_UNKNOWN && (stid == BEGL_S || stid == BEGR_S) {
            allow_s_trunc_end = true;
        }

        if stt == B_ST {
            let y = cm.cfirst[v] as usize;
            let z = cm.cnum[v] as usize;
            let jp_y = j - jmin[y];
            let jp_z = j - jmin[z];
            let kmin = (j - jmax[y]).max(hdmin[z][jp_z as usize]);
            let kmax = jp_y.min(hdmax[z][jp_z as usize]);

            let cur_vec_size = (d + 3) as usize;
            for p in pa.iter_mut().take(cur_vec_size) {
                *p = IMPOSSIBLE;
            }
            for kk in ka.iter_mut().take(cur_vec_size) {
                *kk = -1;
            }
            for p in y_mode_a.iter_mut().take(cur_vec_size) {
                *p = TRMODE_UNKNOWN;
            }
            for p in z_mode_a.iter_mut().take(cur_vec_size) {
                *p = TRMODE_UNKNOWN;
            }

            if v_mode == TRMODE_J {
                if cp9b.jvalid[y] && cp9b.jvalid[z] {
                    let mut k = kmin;
                    while k <= kmax {
                        let jp_yk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jp_yk] && k <= d - hdmin[y][jp_yk] {
                            let kp_z = k - hdmin[z][jp_z as usize];
                            let dp_y = d - hdmin[y][jp_yk];
                            pa[k as usize] = ja[y][jp_yk][(dp_y - k) as usize]
                                + ja[z][jp_z as usize][kp_z as usize];
                            ka[k as usize] = k;
                            y_mode_a[k as usize] = TRMODE_J;
                            z_mode_a[k as usize] = TRMODE_J;
                        }
                        k += 1;
                    }
                }
            } else if v_mode == TRMODE_L {
                if filled_l && cp9b.jvalid[y] && cp9b.lvalid[z] {
                    let mut k = kmin;
                    while k <= kmax {
                        let jp_yk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jp_yk] && k <= d - hdmin[y][jp_yk] {
                            let kp_z = k - hdmin[z][jp_z as usize];
                            let dp_y = d - hdmin[y][jp_yk];
                            pa[k as usize] = ja[y][jp_yk][(dp_y - k) as usize]
                                + la[z][jp_z as usize][kp_z as usize];
                            ka[k as usize] = k;
                            y_mode_a[k as usize] = TRMODE_J;
                            z_mode_a[k as usize] = TRMODE_L;
                        }
                        k += 1;
                    }
                }
                // Two special L cases: entire seq on left, k==0.
                if j >= jmin[y] && j <= jmax[y] {
                    let jp_y2 = (j - jmin[y]) as usize;
                    if d >= hdmin[y][jp_y2] && d <= hdmax[y][jp_y2] {
                        let dp_y = (d - hdmin[y][jp_y2]) as usize;
                        if cp9b.jvalid[y] {
                            pa[(d + 1) as usize] = ja[y][jp_y2][dp_y];
                            ka[(d + 1) as usize] = 0;
                            y_mode_a[(d + 1) as usize] = TRMODE_J;
                            z_mode_a[(d + 1) as usize] = TRMODE_UNKNOWN;
                        }
                        if filled_l && cp9b.lvalid[y] {
                            pa[(d + 2) as usize] = la[y][jp_y2][dp_y];
                            ka[(d + 2) as usize] = 0;
                            y_mode_a[(d + 2) as usize] = TRMODE_L;
                            z_mode_a[(d + 2) as usize] = TRMODE_UNKNOWN;
                        }
                    }
                }
            } else if v_mode == TRMODE_R {
                if filled_r && cp9b.rvalid[y] && cp9b.jvalid[z] {
                    let mut k = kmin;
                    while k <= kmax {
                        let jp_yk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jp_yk] && k <= d - hdmin[y][jp_yk] {
                            let kp_z = k - hdmin[z][jp_z as usize];
                            let dp_y = d - hdmin[y][jp_yk];
                            pa[k as usize] = ra[y][jp_yk][(dp_y - k) as usize]
                                + ja[z][jp_z as usize][kp_z as usize];
                            ka[k as usize] = k;
                            y_mode_a[k as usize] = TRMODE_R;
                            z_mode_a[k as usize] = TRMODE_J;
                        }
                        k += 1;
                    }
                }
                // Two special R cases: entire seq on right, k==d.
                if j >= jmin[z] && j <= jmax[z] {
                    let jp_z2 = (j - jmin[z]) as usize;
                    if d >= hdmin[z][jp_z2] && d <= hdmax[z][jp_z2] {
                        let dp_z = (d - hdmin[z][jp_z2]) as usize;
                        if cp9b.jvalid[z] {
                            pa[(d + 1) as usize] = ja[z][jp_z2][dp_z];
                            ka[(d + 1) as usize] = d;
                            y_mode_a[(d + 1) as usize] = TRMODE_UNKNOWN;
                            z_mode_a[(d + 1) as usize] = TRMODE_J;
                        }
                        if filled_r && cp9b.rvalid[z] {
                            pa[(d + 2) as usize] = ra[z][jp_z2][dp_z];
                            ka[(d + 2) as usize] = d;
                            y_mode_a[(d + 2) as usize] = TRMODE_UNKNOWN;
                            z_mode_a[(d + 2) as usize] = TRMODE_R;
                        }
                    }
                }
            } else if v_mode == TRMODE_T
                && filled_r
                && filled_l
                && cp9b.rvalid[y]
                && cp9b.lvalid[z]
            {
                let kn = kmin.max(1);
                let kx = kmax.min(d);
                let mut k = kn;
                while k <= kx {
                    let jp_yk = (jp_y - k) as usize;
                    if k >= d - hdmax[y][jp_yk] && k <= d - hdmin[y][jp_yk] {
                        let kp_z = k - hdmin[z][jp_z as usize];
                        let dp_y = d - hdmin[y][jp_yk];
                        pa[k as usize] = ra[y][jp_yk][(dp_y - k) as usize]
                            + la[z][jp_z as usize][kp_z as usize];
                        ka[k as usize] = k;
                        y_mode_a[k as usize] = TRMODE_R;
                        z_mode_a[k as usize] = TRMODE_L;
                    }
                    k += 1;
                }
            }

            let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..cur_vec_size])
                .expect("tr_stochastic_parsetree_hb: no valid B_st transitions");
            let y_mode = y_mode_a[choice];
            let z_mode = z_mode_a[choice];
            let k = ka[choice];

            // Push the right fragment onto the stack.
            i_pda.push(j);
            i_pda.push(k);
            i_pda.push(tr.n - 1);
            c_pda.push(z_mode);

            // Attach left start state.
            j -= k;
            d -= k;
            i = j - d + 1;
            let parent = tr.n - 1;
            let idx = tr.add_node_mode(i, j, y as i32, -1, -1, parent, y_mode);
            tr.nxtl[parent as usize] = idx;
            v = y;
            v_mode = y_mode;
        } else if stt == E_ST || stt == EL_ST {
            // Done with left branch; swing to a right fragment or finish.
            if i_pda.is_empty() {
                break;
            }
            let bifparent = i_pda.pop().unwrap();
            d = i_pda.pop().unwrap();
            j = i_pda.pop().unwrap();
            let y_mode = c_pda.pop().unwrap();
            v = tr.state[bifparent as usize] as usize; // B state
            let y = cm.cnum[v] as usize; // right S
            i = j - d + 1;
            let idx = tr.add_node_mode(i, j, y as i32, -1, -1, bifparent, y_mode);
            tr.nxtr[bifparent as usize] = idx;
            v = y;
            v_mode = y_mode;
        } else {
            // v != B, E, EL.
            let mut yoffset: i32;
            let mut y_mode = v_mode;
            let mut b = 0usize;
            let mut b_mode = v_mode;

            if v == 0 {
                // ROOT_S: choose a truncated begin state (same-mode transitions only).
                let do_j = v_mode == TRMODE_J;
                let do_l = v_mode == TRMODE_L;
                let do_r = v_mode == TRMODE_R;
                let do_t = v_mode == TRMODE_T;
                let cur_vec_size = cm.m as usize;
                for p in pa.iter_mut().take(cur_vec_size) {
                    *p = IMPOSSIBLE;
                }
                for yy in 0..cm.m as usize {
                    if j >= jmin[yy] && j <= jmax[yy] {
                        let jp_y = (j - jmin[yy]) as usize;
                        if d >= hdmin[yy][jp_y] && d <= hdmax[yy][jp_y] {
                            let dp_y = (d - hdmin[yy][jp_y]) as usize;
                            let trpenalty = pty[yy];
                            if not_impossible(trpenalty) {
                                if do_j && cp9b.jvalid[yy] {
                                    pa[yy] = trpenalty + ja[yy][jp_y][dp_y];
                                }
                                if filled_l && do_l && cp9b.lvalid[yy] {
                                    pa[yy] = trpenalty + la[yy][jp_y][dp_y];
                                }
                                if filled_r && do_r && cp9b.rvalid[yy] {
                                    pa[yy] = trpenalty + ra[yy][jp_y][dp_y];
                                }
                                if filled_t
                                    && do_t
                                    && cp9b.tvalid[yy]
                                    && cm.sttype[yy] as i32 == B_ST
                                {
                                    pa[yy] = trpenalty + ta[yy][jp_y][dp_y];
                                }
                            }
                        }
                    }
                }
                let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..cur_vec_size])
                    .expect("tr_stochastic_parsetree_hb: no valid local begins");
                b = choice;
                b_mode = v_mode; // S_st can't change mode
                let trpenalty = pty[b];
                fsc += trpenalty;
                yoffset = USED_TRUNC_BEGIN;
                // C sets tr->trpenalty here; the Rust Parsetree does not store it
                // (unused by parsetrees_to_alignment / ParsetreeScore, as with the
                // existing byte-verified tr_cyk/optacc_align_hb tracebacks).
                let _ = local_begin;
            } else {
                // Standard case: v != 0, non-emitter or emitter.
                fsc += get_femission_score_trunc(cm, dsq, lm, rm, v, i, j, v_mode);
                let sd = tr_state_delta(stt);
                let sldelta = tr_state_left_delta(stt);
                let srdelta = tr_state_right_delta(stt);

                if allow_s_trunc_end {
                    yoffset = USED_TRUNC_END;
                    y_mode = TRMODE_UNKNOWN;
                } else if d == 1 && v_mode == TRMODE_L && sldelta == 1 {
                    yoffset = USED_TRUNC_END;
                    y_mode = TRMODE_L;
                } else if d == 1 && v_mode == TRMODE_R && srdelta == 1 {
                    yoffset = USED_TRUNC_END;
                    y_mode = TRMODE_R;
                } else {
                    // Usual case: determine transition.
                    let (vms_sd, vms_sdr, do_j, do_l, do_r);
                    if v_mode == TRMODE_J {
                        vms_sd = sd;
                        vms_sdr = srdelta;
                        do_j = true;
                        do_l = false;
                        do_r = false;
                    } else if v_mode == TRMODE_L {
                        vms_sd = sldelta;
                        vms_sdr = 0;
                        do_j = srdelta == 1;
                        do_l = true;
                        do_r = false;
                    } else {
                        // TRMODE_R
                        vms_sd = srdelta;
                        vms_sdr = srdelta;
                        do_j = sldelta == 1;
                        do_l = false;
                        do_r = true;
                    }

                    let cnum = cm.cnum[v] as usize;
                    let mut jntrans = if do_j { cnum } else { 0 };
                    let mut lntrans = if do_l { cnum } else { 0 };
                    let mut rntrans = if do_r { cnum } else { 0 };
                    let mut jel = false;
                    let mut lel = false;
                    let mut rel = false;
                    if local_end && not_impossible(cm.endsc[v]) {
                        if do_j && cp9b.jvalid[m] {
                            jel = true;
                            jntrans += 1;
                        }
                        if do_l && cp9b.lvalid[m] {
                            lel = true;
                            lntrans += 1;
                        }
                        if do_r && cp9b.rvalid[m] {
                            rel = true;
                            rntrans += 1;
                        }
                    }
                    for p in jpa.iter_mut() {
                        *p = IMPOSSIBLE;
                    }
                    for p in lpa.iter_mut() {
                        *p = IMPOSSIBLE;
                    }
                    for p in rpa.iter_mut() {
                        *p = IMPOSSIBLE;
                    }
                    for yoff in 0..cnum {
                        let yy = cm.cfirst[v] as usize + yoff;
                        if (j - vms_sdr) >= jmin[yy] && (j - vms_sdr) <= jmax[yy] {
                            let jp_y_vms_sdr = (j - jmin[yy] - vms_sdr) as usize;
                            if (d - vms_sd) >= hdmin[yy][jp_y_vms_sdr]
                                && (d - vms_sd) <= hdmax[yy][jp_y_vms_sdr]
                            {
                                let dp = (d - hdmin[yy][jp_y_vms_sdr] - vms_sd) as usize;
                                if do_j && cp9b.jvalid[yy] {
                                    jpa[yoff] = cm.tsc[v][yoff] + ja[yy][jp_y_vms_sdr][dp];
                                }
                                if filled_l && do_l && cp9b.lvalid[yy] {
                                    lpa[yoff] = cm.tsc[v][yoff] + la[yy][jp_y_vms_sdr][dp];
                                }
                                if filled_r && do_r && cp9b.rvalid[yy] {
                                    rpa[yoff] = cm.tsc[v][yoff] + ra[yy][jp_y_vms_sdr][dp];
                                }
                            }
                        }
                    }
                    // EL deck is non-banded: [m][j][d].
                    if jel {
                        jpa[jntrans - 1] = cm.endsc[v] + ja[m][j as usize][d as usize];
                    }
                    if lel {
                        lpa[lntrans - 1] = cm.endsc[v] + la[m][j as usize][d as usize];
                    }
                    if rel {
                        rpa[rntrans - 1] = cm.endsc[v] + ra[m][j as usize][d as usize];
                    }

                    let cur_vec_size = jntrans + lntrans + rntrans;
                    for p in pa.iter_mut().take(cur_vec_size) {
                        *p = IMPOSSIBLE;
                    }
                    let mut p = 0usize;
                    for yoff in 0..jntrans {
                        pa[p] = jpa[yoff];
                        y_mode_a[p] = TRMODE_J;
                        yoffset_a[p] = yoff as i32;
                        p += 1;
                    }
                    for yoff in 0..lntrans {
                        pa[p] = lpa[yoff];
                        y_mode_a[p] = TRMODE_L;
                        yoffset_a[p] = yoff as i32;
                        p += 1;
                    }
                    for yoff in 0..rntrans {
                        pa[p] = rpa[yoff];
                        y_mode_a[p] = TRMODE_R;
                        yoffset_a[p] = yoff as i32;
                        p += 1;
                    }

                    let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..cur_vec_size])
                        .expect("tr_stochastic_parsetree_hb: no valid non-B_st transitions");
                    y_mode = y_mode_a[choice];
                    yoffset = yoffset_a[choice];
                    if (y_mode == TRMODE_J && jel && yoffset == (jntrans as i32 - 1))
                        || (y_mode == TRMODE_L && lel && yoffset == (lntrans as i32 - 1))
                        || (y_mode == TRMODE_R && rel && yoffset == (rntrans as i32 - 1))
                    {
                        yoffset = USED_EL;
                        fsc += cm.endsc[v] + cm.el_selfsc * (d - sd) as f32;
                    } else {
                        fsc += cm.tsc[v][yoffset as usize];
                    }
                }
            }

            // Adjust i and j based on state type and mode.
            match stt {
                x if x == D_ST || x == S_ST => {}
                x if x == MP_ST => {
                    if v_mode == TRMODE_J {
                        i += 1;
                    }
                    if v_mode == TRMODE_L && d > 0 {
                        i += 1;
                    }
                    if v_mode == TRMODE_J {
                        j -= 1;
                    }
                    if v_mode == TRMODE_R && d > 0 {
                        j -= 1;
                    }
                }
                x if x == ML_ST || x == IL_ST => {
                    if v_mode == TRMODE_J {
                        i += 1;
                    }
                    if v_mode == TRMODE_L && d > 0 {
                        i += 1;
                    }
                }
                x if x == MR_ST || x == IR_ST => {
                    if v_mode == TRMODE_J {
                        j -= 1;
                    }
                    if v_mode == TRMODE_R && d > 0 {
                        j -= 1;
                    }
                }
                _ => panic!("tr_stochastic_parsetree_hb: bogus state type {stt}"),
            }
            d = j - i + 1;

            if yoffset == USED_EL || yoffset == USED_TRUNC_END {
                if yoffset == USED_EL {
                    let parent = tr.n - 1;
                    let idx = tr.add_node_mode(i, j, cm.m, -1, -1, parent, y_mode);
                    tr.nxtl[parent as usize] = idx;
                }
                v = m; // now in EL (or acting like it for TRUNC_END)
                v_mode = y_mode;
            } else if yoffset == USED_TRUNC_BEGIN {
                let parent = tr.n - 1;
                let idx = tr.add_node_mode(i, j, b as i32, -1, -1, parent, b_mode);
                tr.nxtl[parent as usize] = idx;
                v = b;
                v_mode = b_mode;
            } else {
                let yy = cm.cfirst[v] as usize + yoffset as usize;
                let parent = tr.n - 1;
                let idx = tr.add_node_mode(i, j, yy as i32, -1, -1, parent, y_mode);
                tr.nxtl[parent as usize] = idx;
                v = yy;
                v_mode = y_mode;
            }
        }
    }

    (tr, parsetree_mode, fsc)
}

/// C `cm_TrStochasticParsetree` (cm_parsetree.c:3303). Non-banded analog of
/// [`tr_stochastic_parsetree_hb`]: samples a parsetree (and its marginal mode,
/// if `preset_mode == TRMODE_UNKNOWN`) from the non-banded truncated Inside
/// matrix `mx` (from [`tr_inside_align`]). Used by `cmalign --sample --nonbanded`
/// (default truncated) and cmbuild --refine --gibbs --nonbanded. `use_local`
/// selects the local vs global truncation-penalty array. Returns
/// `(parsetree, sampled_mode, fsc)`. RNG draw order matches C exactly (mode,
/// then per-state `sample_helper` calls in traceback order).
#[allow(clippy::too_many_arguments)]
pub fn tr_stochastic_parsetree(
    cm: &CM,
    trp: &TrPenalties,
    mx: &TrMx,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    l: i32,
    pass_idx: i32,
    preset_mode: i8,
    use_local: bool,
    r: &mut crate::easel::random::EslRandom,
) -> (Parsetree, i8, f32) {
    let m = cm.m as usize;
    let stride = mx.stride;
    let ncell = mx.ncell;
    let maxconnect = crate::constants::MAXCONNECT as usize;
    let pty = trp.pty_slice(pass_idx, use_local);
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    // Cell index into a J/L/R/T plane for state v at (j,d). EL deck is v==m.
    let cell = |v: usize, j: i32, d: i32| -> usize {
        v * ncell + (j as usize) * stride + d as usize
    };
    let ja = &mx.j;
    let la = &mx.l;
    let ra = &mx.r;
    let ta = &mx.t;

    // C cm_TrFillFromMode: which of L/R/T planes were filled.
    let (filled_l, filled_r, filled_t) = match preset_mode {
        TRMODE_J => (false, false, false),
        TRMODE_L => (true, false, false),
        TRMODE_R => (false, true, false),
        _ => (true, true, true),
    };

    // Multipurpose vectors (C vec_size = max(L+3, max(M, 3*MAXCONNECT+1))).
    let vec_size = (l + 3).max((cm.m).max(3 * maxconnect as i32 + 1)) as usize;
    let mut pa = vec![IMPOSSIBLE; vec_size];
    let mut y_mode_a = vec![TRMODE_UNKNOWN; vec_size];
    let mut z_mode_a = vec![TRMODE_UNKNOWN; vec_size];
    let mut ka = vec![0i32; vec_size];
    let mut yoffset_a = vec![0i32; vec_size];
    let mut jpa = vec![IMPOSSIBLE; maxconnect + 1];
    let mut lpa = vec![IMPOSSIBLE; maxconnect + 1];
    let mut rpa = vec![IMPOSSIBLE; maxconnect + 1];

    // C 3381-3395: sample the marginal mode if preset_mode is UNKNOWN, from the
    // four corner cells Xalpha[0][L][L]. (C hardcodes validA[0..3]=TRUE, but
    // sample_helper re-derives validity from NOT_IMPOSSIBLE, so IMPOSSIBLE
    // corners are simply skipped.)
    let mut parsetree_mode = preset_mode;
    if preset_mode == TRMODE_UNKNOWN {
        pa[0] = ja[cell(0, l, l)];
        pa[1] = la[cell(0, l, l)];
        pa[2] = ra[cell(0, l, l)];
        pa[3] = ta[cell(0, l, l)];
        let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..4])
            .expect("tr_stochastic_parsetree: no valid alignment modes");
        parsetree_mode = match choice {
            0 => TRMODE_J,
            1 => TRMODE_L,
            2 => TRMODE_R,
            _ => TRMODE_T,
        };
    }

    // Init parse tree with the root S state (mode = parsetree_mode).
    let mut tr = Parsetree::new(100);
    tr.is_std = false;
    tr.pass_idx = pass_idx;
    tr.add_node_mode(1, l, 0, -1, -1, -1, parsetree_mode);

    let mut i_pda: Vec<i32> = Vec::new();
    let mut c_pda: Vec<i8> = Vec::new();

    let mut v: usize = 0;
    let mut j = l;
    let mut d = l;
    let mut i = 1;
    let mut v_mode = parsetree_mode;
    let mut fsc = 0.0f32;

    loop {
        let stt = cm.sttype[v] as i32;
        let stid = cm.stid[v] as i32;
        // Super-special truncated-end case (d==0, mode UNKNOWN, BEGL_S/BEGR_S).
        let mut allow_s_trunc_end = false;
        if d == 0 && v_mode == TRMODE_UNKNOWN && (stid == BEGL_S || stid == BEGR_S) {
            allow_s_trunc_end = true;
        }

        if stt == B_ST {
            let y = cm.cfirst[v] as usize;
            let z = cm.cnum[v] as usize;
            let cur_vec_size = (d + 3) as usize;
            for p in pa.iter_mut().take(cur_vec_size) {
                *p = IMPOSSIBLE;
            }
            for kk in ka.iter_mut().take(cur_vec_size) {
                *kk = -1;
            }
            for p in y_mode_a.iter_mut().take(cur_vec_size) {
                *p = TRMODE_UNKNOWN;
            }
            for p in z_mode_a.iter_mut().take(cur_vec_size) {
                *p = TRMODE_UNKNOWN;
            }

            if v_mode == TRMODE_J {
                // v is J: y and z both J.
                for k in 0..=d {
                    pa[k as usize] = ja[cell(y, j - k, d - k)] + ja[cell(z, j, k)];
                    ka[k as usize] = k;
                    y_mode_a[k as usize] = TRMODE_J;
                    z_mode_a[k as usize] = TRMODE_J;
                }
            } else if v_mode == TRMODE_L && filled_l {
                // v is L: y is J or L, z is L.
                for k in 0..=d {
                    pa[k as usize] = ja[cell(y, j - k, d - k)] + la[cell(z, j, k)];
                    ka[k as usize] = k;
                    y_mode_a[k as usize] = TRMODE_J;
                    z_mode_a[k as usize] = TRMODE_L;
                }
                // Two special L cases: entire seq on left, k==0.
                pa[(d + 1) as usize] = ja[cell(y, j, d)];
                ka[(d + 1) as usize] = 0;
                y_mode_a[(d + 1) as usize] = TRMODE_J;
                z_mode_a[(d + 1) as usize] = TRMODE_UNKNOWN;

                pa[(d + 2) as usize] = la[cell(y, j, d)];
                ka[(d + 2) as usize] = 0;
                y_mode_a[(d + 2) as usize] = TRMODE_L;
                z_mode_a[(d + 2) as usize] = TRMODE_UNKNOWN;
            } else if v_mode == TRMODE_R && filled_r {
                // v is R: y is R, z is J or R.
                for k in 0..=d {
                    pa[k as usize] = ra[cell(y, j - k, d - k)] + ja[cell(z, j, k)];
                    ka[k as usize] = k;
                    y_mode_a[k as usize] = TRMODE_R;
                    z_mode_a[k as usize] = TRMODE_J;
                }
                // Two special R cases: entire seq on right, k==d.
                pa[(d + 1) as usize] = ja[cell(z, j, d)];
                ka[(d + 1) as usize] = d;
                y_mode_a[(d + 1) as usize] = TRMODE_UNKNOWN;
                z_mode_a[(d + 1) as usize] = TRMODE_J;

                pa[(d + 2) as usize] = ra[cell(z, j, d)];
                ka[(d + 2) as usize] = d;
                y_mode_a[(d + 2) as usize] = TRMODE_UNKNOWN;
                z_mode_a[(d + 2) as usize] = TRMODE_R;
            } else if v_mode == TRMODE_T && filled_r && filled_l {
                // v is T: y is R, z is L. k in 1..d.
                for k in 1..d {
                    pa[k as usize] = ra[cell(y, j - k, d - k)] + la[cell(z, j, k)];
                    ka[k as usize] = k;
                    y_mode_a[k as usize] = TRMODE_R;
                    z_mode_a[k as usize] = TRMODE_L;
                }
            }

            let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..cur_vec_size])
                .expect("tr_stochastic_parsetree: no valid B_st transitions");
            let y_mode = y_mode_a[choice];
            let z_mode = z_mode_a[choice];
            let k = ka[choice];

            i_pda.push(j);
            i_pda.push(k);
            i_pda.push(tr.n - 1);
            c_pda.push(z_mode);

            j -= k;
            d -= k;
            i = j - d + 1;
            let parent = tr.n - 1;
            let idx = tr.add_node_mode(i, j, y as i32, -1, -1, parent, y_mode);
            tr.nxtl[parent as usize] = idx;
            v = y;
            v_mode = y_mode;
        } else if stt == E_ST || stt == EL_ST {
            // Done with left branch; swing to a right fragment or finish.
            if i_pda.is_empty() {
                break;
            }
            let bifparent = i_pda.pop().unwrap();
            d = i_pda.pop().unwrap();
            j = i_pda.pop().unwrap();
            let y_mode = c_pda.pop().unwrap();
            v = tr.state[bifparent as usize] as usize; // B state
            let y = cm.cnum[v] as usize; // right S
            i = j - d + 1;
            let idx = tr.add_node_mode(i, j, y as i32, -1, -1, bifparent, y_mode);
            tr.nxtr[bifparent as usize] = idx;
            v = y;
            v_mode = y_mode;
        } else {
            // v != B, E, EL.
            let mut yoffset: i32;
            let mut y_mode = v_mode;
            let mut b = 0usize;
            let mut b_mode = v_mode;

            if v == 0 {
                // ROOT_S: choose a truncated begin state (same-mode transitions only).
                let do_j = v_mode == TRMODE_J;
                let do_l = v_mode == TRMODE_L;
                let do_r = v_mode == TRMODE_R;
                let do_t = v_mode == TRMODE_T;
                let cur_vec_size = cm.m as usize;
                for p in pa.iter_mut().take(cur_vec_size) {
                    *p = IMPOSSIBLE;
                }
                for yy in 0..cm.m as usize {
                    let trpenalty = pty[yy];
                    if not_impossible(trpenalty) {
                        if do_j {
                            pa[yy] = trpenalty + ja[cell(yy, j, d)];
                        }
                        if filled_l && do_l {
                            pa[yy] = trpenalty + la[cell(yy, j, d)];
                        }
                        if filled_r && do_r {
                            pa[yy] = trpenalty + ra[cell(yy, j, d)];
                        }
                        if filled_t && do_t && cm.sttype[yy] as i32 == B_ST {
                            pa[yy] = trpenalty + ta[cell(yy, j, d)];
                        }
                    }
                }
                let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..cur_vec_size])
                    .expect("tr_stochastic_parsetree: no valid local begins");
                b = choice;
                b_mode = v_mode; // S_st can't change mode
                let trpenalty = pty[b];
                fsc += trpenalty;
                yoffset = USED_TRUNC_BEGIN;
                tr.trpenalty = trpenalty; // C sets tr->trpenalty (cm_parsetree.c:3581)
            } else {
                // Standard case: v != 0, non-emitter or emitter.
                fsc += get_femission_score_trunc(cm, dsq, lm, rm, v, i, j, v_mode);
                let sd = tr_state_delta(stt);
                let sldelta = tr_state_left_delta(stt);
                let srdelta = tr_state_right_delta(stt);

                if allow_s_trunc_end {
                    yoffset = USED_TRUNC_END;
                    y_mode = TRMODE_UNKNOWN;
                } else if d == 1 && v_mode == TRMODE_L && sldelta == 1 {
                    yoffset = USED_TRUNC_END;
                    y_mode = TRMODE_L;
                } else if d == 1 && v_mode == TRMODE_R && srdelta == 1 {
                    yoffset = USED_TRUNC_END;
                    y_mode = TRMODE_R;
                } else {
                    let (vms_sd, vms_sdr, do_j, do_l, do_r);
                    if v_mode == TRMODE_J {
                        vms_sd = sd;
                        vms_sdr = srdelta;
                        do_j = true;
                        do_l = false;
                        do_r = false;
                    } else if v_mode == TRMODE_L {
                        vms_sd = sldelta;
                        vms_sdr = 0;
                        do_j = srdelta == 1;
                        do_l = true;
                        do_r = false;
                    } else {
                        // TRMODE_R
                        vms_sd = srdelta;
                        vms_sdr = srdelta;
                        do_j = sldelta == 1;
                        do_l = false;
                        do_r = true;
                    }

                    let cnum = cm.cnum[v] as usize;
                    let mut jntrans = if do_j { cnum } else { 0 };
                    let mut lntrans = if do_l { cnum } else { 0 };
                    let mut rntrans = if do_r { cnum } else { 0 };
                    let mut jel = false;
                    let mut lel = false;
                    let mut rel = false;
                    if local_end && not_impossible(cm.endsc[v]) {
                        if do_j {
                            jel = true;
                            jntrans += 1;
                        }
                        if do_l {
                            lel = true;
                            lntrans += 1;
                        }
                        if do_r {
                            rel = true;
                            rntrans += 1;
                        }
                    }
                    for p in jpa.iter_mut() {
                        *p = IMPOSSIBLE;
                    }
                    for p in lpa.iter_mut() {
                        *p = IMPOSSIBLE;
                    }
                    for p in rpa.iter_mut() {
                        *p = IMPOSSIBLE;
                    }
                    for yoff in 0..cnum {
                        let yy = cm.cfirst[v] as usize + yoff;
                        if do_j {
                            jpa[yoff] =
                                cm.tsc[v][yoff] + ja[cell(yy, j - vms_sdr, d - vms_sd)];
                        }
                        if filled_l && do_l {
                            lpa[yoff] =
                                cm.tsc[v][yoff] + la[cell(yy, j - vms_sdr, d - vms_sd)];
                        }
                        if filled_r && do_r {
                            rpa[yoff] =
                                cm.tsc[v][yoff] + ra[cell(yy, j - vms_sdr, d - vms_sd)];
                        }
                    }
                    // EL deck (v==m).
                    if jel {
                        jpa[jntrans - 1] = cm.endsc[v] + ja[cell(m, j, d)];
                    }
                    if lel {
                        lpa[lntrans - 1] = cm.endsc[v] + la[cell(m, j, d)];
                    }
                    if rel {
                        rpa[rntrans - 1] = cm.endsc[v] + ra[cell(m, j, d)];
                    }

                    let cur_vec_size = jntrans + lntrans + rntrans;
                    for p in pa.iter_mut().take(cur_vec_size) {
                        *p = IMPOSSIBLE;
                    }
                    let mut p = 0usize;
                    for yoff in 0..jntrans {
                        pa[p] = jpa[yoff];
                        y_mode_a[p] = TRMODE_J;
                        yoffset_a[p] = yoff as i32;
                        p += 1;
                    }
                    for yoff in 0..lntrans {
                        pa[p] = lpa[yoff];
                        y_mode_a[p] = TRMODE_L;
                        yoffset_a[p] = yoff as i32;
                        p += 1;
                    }
                    for yoff in 0..rntrans {
                        pa[p] = rpa[yoff];
                        y_mode_a[p] = TRMODE_R;
                        yoffset_a[p] = yoff as i32;
                        p += 1;
                    }

                    let choice = crate::cm_dpalign::sample_helper(r, &mut pa[..cur_vec_size])
                        .expect("tr_stochastic_parsetree: no valid non-B_st transitions");
                    y_mode = y_mode_a[choice];
                    yoffset = yoffset_a[choice];
                    if (y_mode == TRMODE_J && jel && yoffset == (jntrans as i32 - 1))
                        || (y_mode == TRMODE_L && lel && yoffset == (lntrans as i32 - 1))
                        || (y_mode == TRMODE_R && rel && yoffset == (rntrans as i32 - 1))
                    {
                        yoffset = USED_EL;
                        fsc += cm.endsc[v] + cm.el_selfsc * (d - sd) as f32;
                    } else {
                        fsc += cm.tsc[v][yoffset as usize];
                    }
                }
            }

            // Adjust i and j based on state type and mode (C 3685-3712).
            match stt {
                x if x == D_ST || x == S_ST => {}
                x if x == MP_ST => {
                    if v_mode == TRMODE_J {
                        i += 1;
                    }
                    if v_mode == TRMODE_L && d > 0 {
                        i += 1;
                    }
                    if v_mode == TRMODE_J {
                        j -= 1;
                    }
                    if v_mode == TRMODE_R && d > 0 {
                        j -= 1;
                    }
                }
                x if x == ML_ST || x == IL_ST => {
                    if v_mode == TRMODE_J {
                        i += 1;
                    }
                    if v_mode == TRMODE_L && d > 0 {
                        i += 1;
                    }
                }
                x if x == MR_ST || x == IR_ST => {
                    if v_mode == TRMODE_J {
                        j -= 1;
                    }
                    if v_mode == TRMODE_R && d > 0 {
                        j -= 1;
                    }
                }
                _ => panic!("tr_stochastic_parsetree: bogus state type {stt}"),
            }
            d = j - i + 1;

            if yoffset == USED_EL || yoffset == USED_TRUNC_END {
                if yoffset == USED_EL {
                    let parent = tr.n - 1;
                    let idx = tr.add_node_mode(i, j, cm.m, -1, -1, parent, y_mode);
                    tr.nxtl[parent as usize] = idx;
                }
                v = m; // now in EL (or acting like it for TRUNC_END)
                v_mode = y_mode;
            } else if yoffset == USED_TRUNC_BEGIN {
                let parent = tr.n - 1;
                let idx = tr.add_node_mode(i, j, b as i32, -1, -1, parent, b_mode);
                tr.nxtl[parent as usize] = idx;
                v = b;
                v_mode = b_mode;
            } else {
                let yy = cm.cfirst[v] as usize + yoffset as usize;
                let parent = tr.n - 1;
                let idx = tr.add_node_mode(i, j, yy as i32, -1, -1, parent, y_mode);
                tr.nxtl[parent as usize] = idx;
                v = yy;
                v_mode = y_mode;
            }
        }
    }

    (tr, parsetree_mode, fsc)
}
