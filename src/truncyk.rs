//! Faithful port of C `truncyk.c` — the truncated divide-and-conquer *exact* CYK
//! alignment engine (`TrCYK_DnC`), used by `cmbuild --refine --nonbanded` (the
//! default, truncated case: `cm_alndata.c:379` `TrCYK_DnC(cm,dsq,L,0,1,L,pass_idx,
//! FALSE,&tr)`).
//!
//! This is the truncated analog of [`crate::cm_dpsmall`] (`CYKDivideAndConquer`),
//! adding the four marginal alignment modes J/L/R/T (Joint / Left-truncated /
//! Right-truncated / Terminal-bifurcation). Structure mirrors C `truncyk.c` exactly:
//!   * [`tr_cyk_dnc`]            — C `TrCYK_DnC`            (truncyk.c:322)
//!   * `tr_generic_splitter`    — C `tr_generic_splitter`  (truncyk.c:501)
//!   * `tr_wedge_splitter`      — C `tr_wedge_splitter`    (truncyk.c:830)
//!   * `tr_v_splitter`          — C `tr_v_splitter`        (truncyk.c:1025)
//!   * [`tr_inside`]            — vjd truncated CYK Inside  (truncyk.c:1259)
//!   * `tr_outside`             — vjd truncated CYK Outside (truncyk.c:2184)
//!   * `tr_vinside`/`tr_voutside` — vji truncated engines   (truncyk.c:2730/3384)
//!   * `tr_insideT`/`tr_vinsideT` — fill+traceback          (truncyk.c:3858/4083)
//!
//! Marginal-mode integer encoding matches C `truncyk.c` raw ints exactly:
//! `T=0, R=1, L=2, J=3` (identical to [`crate::cm_trunc`]'s `TRMODE_*`).
//!
//! We use this port for the `do_1p0 == FALSE` path only (the cmbuild refine call);
//! the `do_1p0 == TRUE` "reproduce v1.0 behavior" branch (trcyk benchmark) and its
//! `SetMarginalScores_reproduce_i27` buggy marginal scores are not ported. Marginal
//! emission scores come from the same `lm`/`rm` (`cm.lmesc`/`cm.rmesc`, full
//! augmented alphabet 0..Kp) used by the byte-verified [`crate::cm_trunc::tr_cyk_align`].
//!
//! Memory note (same as `cm_dpsmall`): C's `deckpool_s`/`touch`/`nends` machinery is
//! a pure deck-reuse memory optimization with no effect on computed values or the
//! output parsetree. We use straightforward per-deck `Vec` allocations; the D&C
//! split structure that bounds peak memory is ported faithfully.

// WIP: this module is being ported function-by-function (tr_inside done first).
// Items not yet wired into the dispatch are allowed dead until the engine is complete.
#![allow(dead_code)]

use crate::cm::CM;
use crate::constants::{B_ST, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST};
use crate::parsetree::Parsetree;

const IMPOSSIBLE: f32 = -1.0e36;
const USED_EL: i32 = 102;
const USED_LOCAL_BEGIN: i32 = 101;
const K: usize = 4;

// Marginal modes (C truncyk.c raw ints; == cm_trunc::TRMODE_*).
const MODE_T: i32 = 0;
const MODE_R: i32 = 1;
const MODE_L: i32 = 2;
const MODE_J: i32 = 3;

#[inline]
fn not_impossible(x: f32) -> bool {
    x > -0.5e36
}

/// C singlet emission with degenerate averaging (`esc[dsq]` or `esl_abc_FAvgScore`).
/// Reused convention from the byte-verified `cm_dpsmall::sing_sc`.
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

/// C pair emission with degenerate averaging (`DegeneratePairScore`).
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

// ---- vjd deck management (coords [v][j][d], j absolute i0-1..j0, d 0..j-i0+1) ----
type VjdDeck = Vec<Vec<f32>>;
type VjdShadow = Vec<Vec<i32>>;

/// C `alloc_vjd_deck(L,i0,j0)`: rows 0..=L, row j (i0-1..=j0) has (j-i0+2) cells.
fn new_vjd_deck(l: i32, i0: i32, j0: i32, init: f32) -> VjdDeck {
    let mut a: VjdDeck = vec![Vec::new(); (l + 1) as usize];
    for j in (i0 - 1)..=j0 {
        a[j as usize] = vec![init; (j - i0 + 2) as usize];
    }
    a
}
fn new_vjd_shadow(l: i32, i0: i32, j0: i32, init: i32) -> VjdShadow {
    let mut a: VjdShadow = vec![Vec::new(); (l + 1) as usize];
    for j in (i0 - 1)..=j0 {
        a[j as usize] = vec![init; (j - i0 + 2) as usize];
    }
    a
}

/// The four J/L/R/T alpha deck-sets (C `AlphaMats_t`), each indexed `[v]` over
/// `0..=M` with `None` = unallocated deck.
pub(crate) struct AlphaMats {
    pub j: Vec<Option<VjdDeck>>,
    pub l: Vec<Option<VjdDeck>>,
    pub r: Vec<Option<VjdDeck>>,
    pub t: Vec<Option<VjdDeck>>,
}
impl AlphaMats {
    fn new(m: usize) -> Self {
        AlphaMats {
            j: (0..=m).map(|_| None).collect(),
            l: (0..=m).map(|_| None).collect(),
            r: (0..=m).map(|_| None).collect(),
            t: (0..=m).map(|_| None).collect(),
        }
    }
}

/// Shadow (traceback) decks (C `ShadowMats_t`): J/L/R/T store yoffset or bifurcation
/// `k`; `lmode`/`rmode` store the marginal mode chosen for the L/R deck cell.
pub(crate) struct ShadowMats {
    pub j: Vec<Option<VjdShadow>>,
    pub l: Vec<Option<VjdShadow>>,
    pub r: Vec<Option<VjdShadow>>,
    pub t: Vec<Option<VjdShadow>>,
    pub lmode: Vec<Option<VjdShadow>>,
    pub rmode: Vec<Option<VjdShadow>>,
}
impl ShadowMats {
    fn new(m: usize) -> Self {
        ShadowMats {
            j: (0..m).map(|_| None).collect(),
            l: (0..m).map(|_| None).collect(),
            r: (0..m).map(|_| None).collect(),
            t: (0..m).map(|_| None).collect(),
            lmode: (0..m).map(|_| None).collect(),
            rmode: (0..m).map(|_| None).collect(),
        }
    }
}

/// Best local-hit result returned by [`tr_inside`] (C `ret_mode`/`ret_v`/`ret_i`/`ret_j`
/// plus the returned score `r_sc`).
pub(crate) struct TrInsideRet {
    pub sc: f32,
    pub mode: i32,
    pub v: i32,
    pub i: i32,
    pub j: i32,
}

// ================================================================
// tr_inside() — C truncyk.c:1259
// Fills alpha J/L/R/T[vroot..vend] (vjd) into `alpha`. If `shadow` is Some, fills
// the six shadow decks for traceback. Returns the best local-hit (r_sc,mode,v,i,j).
//
// `alpha` may already hold decks from a previous call (C `arg_alpha`): this lets
// tr_generic_splitter fill w's and y's subtrees into one shared AlphaMats.
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_inside(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    vroot: usize,
    vend: usize,
    i0: i32,
    j0: i32,
    _do_full: bool,
    allow_begin: bool,
    _r_allow_j: bool,
    _r_allow_l: bool,
    _r_allow_r: bool,
    len_correx: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    alpha: &mut AlphaMats,
    mut shadow: Option<&mut ShadowMats>,
) -> TrInsideRet {
    let want_shadow = shadow.is_some();
    let m = cm.m as usize;

    // Initialization (C truncyk.c:1306-1315)
    let mut r_v: i32 = -1;
    let mut r_i: i32 = i0;
    let mut r_j: i32 = j0;
    let mut r_mode: i32 = MODE_J;
    let mut r_sc: f32 = IMPOSSIBLE;
    let w = j0 - i0 + 1;
    let p1_tmp = l as f32 / (l as f32 + 2.0);
    let p2 = (p1_tmp).log2(); // sreLOG2(p1)
    let p1 = 2.0 * (1.0 - p1_tmp).log2(); // 2*sreLOG2(1-p1)

    // Main recursion: v = vend .. vroot
    for v in (vroot..=vend).rev() {
        let stt = cm.sttype[v] as i32;

        // E state: shared 'end' deck. J=L=R = (end[j][0]=0, else IMPOSSIBLE); T unused.
        if stt == E_ST {
            let mut end = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
            for jp in 0..=w {
                let j = (i0 + jp - 1) as usize;
                end[j][0] = 0.0;
            }
            alpha.j[v] = Some(end.clone());
            alpha.l[v] = Some(end.clone());
            alpha.r[v] = Some(end);
            continue;
        }

        // Allocate this state's J/L/R (and T for B) decks + shadow decks.
        let mut dj = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        let mut dl = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        let mut dr = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        let mut dt = if stt == B_ST {
            new_vjd_deck(l, i0, j0, IMPOSSIBLE)
        } else {
            Vec::new()
        };
        let (mut sh_j, mut sh_l, mut sh_r, mut sh_t, mut sh_lm, mut sh_rm) = if want_shadow {
            (
                new_vjd_shadow(l, i0, j0, 0),
                new_vjd_shadow(l, i0, j0, 0),
                new_vjd_shadow(l, i0, j0, 0),
                if stt == B_ST {
                    new_vjd_shadow(l, i0, j0, 0)
                } else {
                    Vec::new()
                },
                new_vjd_shadow(l, i0, j0, 0),
                new_vjd_shadow(l, i0, j0, 0),
            )
        } else {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };

        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let endsc_v = cm.endsc[v];
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];
        let elself = cm.el_selfsc;

        if stt == D_ST || stt == S_ST {
            // C truncyk.c:1447-1502
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                for d in 0..=(jp as usize) {
                    let mut vj = endsc_v + elself * (d as f32);
                    let mut vl = IMPOSSIBLE;
                    let mut vr = IMPOSSIBLE;
                    let mut shj = USED_EL;
                    let mut shl = USED_EL;
                    let mut shr = USED_EL;
                    let mut lmode = 0;
                    let mut rmode = 0;
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        let scj = alpha.j[ch].as_ref().unwrap()[j][d] + tsc_v[yoffset];
                        if scj > vj {
                            vj = scj;
                            shj = yoffset as i32;
                        }
                        let scl = alpha.l[ch].as_ref().unwrap()[j][d] + tsc_v[yoffset];
                        if scl > vl {
                            vl = scl;
                            shl = yoffset as i32;
                            lmode = MODE_L;
                        }
                        let scr = alpha.r[ch].as_ref().unwrap()[j][d] + tsc_v[yoffset];
                        if scr > vr {
                            vr = scr;
                            shr = yoffset as i32;
                            rmode = MODE_R;
                        }
                    }
                    if d == 0 {
                        vl = IMPOSSIBLE;
                        vr = IMPOSSIBLE;
                    }
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[j][d] = vj;
                    dl[j][d] = vl;
                    dr[j][d] = vr;
                    if want_shadow {
                        sh_j[j][d] = shj;
                        sh_l[j][d] = shl;
                        sh_r[j][d] = shr;
                        sh_lm[j][d] = lmode;
                        sh_rm[j][d] = rmode;
                    }
                }
            }
        } else if stt == B_ST {
            // C truncyk.c:1503-1624
            let y = cfirst;
            let z = cnum; // right child state index for B
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                for d in 0..=(jp as usize) {
                    let mut allow_l_exit = false;
                    let mut allow_r_exit = false;
                    let mut allow_j_exit = false;

                    let ayj = alpha.j[y].as_ref().unwrap();
                    let azj = alpha.j[z].as_ref().unwrap();
                    let ayl = alpha.l[y].as_ref().unwrap();
                    let azl = alpha.l[z].as_ref().unwrap();
                    let ayr = alpha.r[y].as_ref().unwrap();
                    let azr = alpha.r[z].as_ref().unwrap();

                    let mut vj = ayj[j][d] + azj[j][0];
                    let mut vl = ayl[j][d];
                    let mut vr = azr[j][d];
                    let mut shj = 0i32;
                    let mut shl = 0i32;
                    let mut shr = d as i32;
                    let mut lmode = MODE_L;
                    let mut rmode = MODE_R;

                    // L: exit into left child in J mode (whole d on left)
                    let sc = ayj[j][d];
                    if sc > vl {
                        vl = sc;
                        shl = 0;
                        lmode = MODE_J;
                    }

                    for k in 1..=d {
                        let scj = ayj[j - k][d - k] + azj[j][k];
                        if scj > vj {
                            vj = scj;
                            shj = k as i32;
                            allow_j_exit = k != d;
                        }
                        let scl = ayj[j - k][d - k] + azl[j][k];
                        if scl > vl {
                            vl = scl;
                            shl = k as i32;
                            lmode = MODE_J;
                            allow_l_exit = true;
                        }
                    }
                    // R base: whole d on right child (J mode)
                    let sc = azj[j][d];
                    if sc > vr {
                        vr = sc;
                        shr = d as i32;
                        rmode = MODE_J;
                    }
                    for k in 0..d {
                        let scr = ayr[j - k][d - k] + azj[j][k];
                        if scr > vr {
                            vr = scr;
                            shr = k as i32;
                            rmode = MODE_J;
                            allow_r_exit = true;
                        }
                    }

                    if d == 0 {
                        vl = IMPOSSIBLE;
                        vr = IMPOSSIBLE;
                    }

                    // T deck (terminal bifurcation): R(y) + L(z)
                    let mut vt = IMPOSSIBLE;
                    let mut sht = 0i32;
                    if d >= 2 {
                        vt = ayr[j - 1][d - 1] + azl[j][1];
                        sht = 1;
                        for k in 2..d {
                            let sc = ayr[j - k][d - k] + azl[j][k];
                            if sc > vt {
                                vt = sc;
                                sht = k as i32;
                            }
                        }
                    }

                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }

                    dj[j][d] = vj;
                    dl[j][d] = vl;
                    dr[j][d] = vr;
                    dt[j][d] = vt;
                    if want_shadow {
                        sh_j[j][d] = shj;
                        sh_l[j][d] = shl;
                        sh_r[j][d] = shr;
                        sh_t[j][d] = sht;
                        sh_lm[j][d] = lmode;
                        sh_rm[j][d] = rmode;
                    }

                    // Local-begin (marginal B exit) bookkeeping.
                    if allow_begin {
                        if !len_correx {
                            if vj > r_sc && allow_j_exit {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj;
                            }
                            if vl > r_sc && allow_l_exit {
                                r_mode = MODE_L;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vl;
                            }
                            if vr > r_sc && allow_r_exit {
                                r_mode = MODE_R;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vr;
                            }
                            if vt > r_sc {
                                r_mode = MODE_T;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vt;
                            }
                        } else {
                            let psc = p1 + (l - d as i32) as f32 * p2;
                            if vj + psc > r_sc && allow_j_exit {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj + psc;
                            }
                            if vl + psc > r_sc && allow_l_exit {
                                r_mode = MODE_L;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vl + psc;
                            }
                            if vr + psc > r_sc && allow_r_exit {
                                r_mode = MODE_R;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vr + psc;
                            }
                            if vt + psc > r_sc {
                                r_mode = MODE_T;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vt + psc;
                            }
                        }
                    }
                }
            }
        } else if stt == MP_ST {
            // C truncyk.c:1625-1716
            let lmv = &lm[v];
            let rmv = &rm[v];
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                dj[j][0] = IMPOSSIBLE;
                dl[j][0] = IMPOSSIBLE;
                dr[j][0] = IMPOSSIBLE;
                if jp > 0 {
                    dj[j][1] = IMPOSSIBLE;
                    dl[j][1] = lmv[dsq[j] as usize];
                    dr[j][1] = rmv[dsq[j] as usize];
                    if want_shadow {
                        sh_l[j][1] = USED_EL;
                        sh_r[j][1] = USED_EL;
                    }
                }
                for d in 2..=(jp as usize) {
                    let mut vj = endsc_v + elself * ((d - 2) as f32);
                    let mut vl = IMPOSSIBLE;
                    let mut vr = IMPOSSIBLE;
                    let mut shj = USED_EL;
                    let mut shl = 0i32;
                    let mut shr = 0i32;
                    let mut lmode = 0;
                    let mut rmode = 0;
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        let scj = alpha.j[ch].as_ref().unwrap()[j - 1][d - 2] + tsc_v[yoffset];
                        if scj > vj {
                            vj = scj;
                            shj = yoffset as i32;
                        }
                        let scl1 = alpha.j[ch].as_ref().unwrap()[j][d - 1] + tsc_v[yoffset];
                        if scl1 > vl {
                            vl = scl1;
                            shl = yoffset as i32;
                            lmode = MODE_J;
                        }
                        let scl2 = alpha.l[ch].as_ref().unwrap()[j][d - 1] + tsc_v[yoffset];
                        if scl2 > vl {
                            vl = scl2;
                            shl = yoffset as i32;
                            lmode = MODE_L;
                        }
                        let scr1 = alpha.j[ch].as_ref().unwrap()[j - 1][d - 1] + tsc_v[yoffset];
                        if scr1 > vr {
                            vr = scr1;
                            shr = yoffset as i32;
                            rmode = MODE_J;
                        }
                        let scr2 = alpha.r[ch].as_ref().unwrap()[j - 1][d - 1] + tsc_v[yoffset];
                        if scr2 > vr {
                            vr = scr2;
                            shr = yoffset as i32;
                            rmode = MODE_R;
                        }
                    }
                    let i = j - d + 1;
                    vj += pair_sc(esc_v, dsq[i], dsq[j]);
                    vl += lmv[dsq[i] as usize];
                    vr += rmv[dsq[j] as usize];
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[j][d] = vj;
                    dl[j][d] = vl;
                    dr[j][d] = vr;
                    if want_shadow {
                        sh_j[j][d] = shj;
                        sh_l[j][d] = shl;
                        sh_r[j][d] = shr;
                        sh_lm[j][d] = lmode;
                        sh_rm[j][d] = rmode;
                    }
                }
                // local-begin: d = 1..jp
                if allow_begin {
                    for d in 1..=(jp as usize) {
                        let vj = dj[j][d];
                        let vl = dl[j][d];
                        let vr = dr[j][d];
                        if !len_correx {
                            if vj > r_sc {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj;
                            }
                            if vl > r_sc {
                                r_mode = MODE_L;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vl;
                            }
                            if vr > r_sc {
                                r_mode = MODE_R;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vr;
                            }
                        } else {
                            let psc = p1 + (l - d as i32) as f32 * p2;
                            if vj + psc > r_sc {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj + psc;
                            }
                            if vl + psc > r_sc {
                                r_mode = MODE_L;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vl + psc;
                            }
                            if vr + psc > r_sc {
                                r_mode = MODE_R;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vr + psc;
                            }
                        }
                    }
                }
            }
        } else if stt == IL_ST || stt == ML_ST {
            // C truncyk.c:1717-1865
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                dj[j][0] = IMPOSSIBLE;
                dl[j][0] = IMPOSSIBLE;
                dr[j][0] = IMPOSSIBLE;
                for d in 1..=(jp as usize) {
                    let mut vj = endsc_v + elself * ((d - 1) as f32);
                    let mut vl = if d == 1 { 0.0 } else { IMPOSSIBLE };
                    let mut vr = IMPOSSIBLE;
                    let mut shj = USED_EL;
                    let mut shl = USED_EL;
                    let mut shr = 0i32;
                    let mut lmode = MODE_J;
                    let mut rmode = 0;
                    // First loop: J and L (depend on d-1)
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        let cvj = if ch == v { dj[j][d - 1] } else { alpha.j[ch].as_ref().unwrap()[j][d - 1] };
                        let scj = cvj + tsc_v[yoffset];
                        if scj > vj {
                            vj = scj;
                            shj = yoffset as i32;
                        }
                        if d > 1 {
                            let cvl = if ch == v { dl[j][d - 1] } else { alpha.l[ch].as_ref().unwrap()[j][d - 1] };
                            let scl = cvl + tsc_v[yoffset];
                            if scl > vl {
                                vl = scl;
                                shl = yoffset as i32;
                                lmode = MODE_L;
                            }
                        }
                    }
                    // Second loop: R (depends on fully-computed J cell alpha[v][j][d]).
                    // C truncyk.c:1759-1774. Separate loop because R needs the finished
                    // J cell (for self-looping IL, yoffset 0). Both reads at [j][d].
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        // scr1: alpha[y+yoffset][j][d] (J deck); self-loop -> J accum vj. Rmode=3.
                        let cvj = if ch == v { vj } else { alpha.j[ch].as_ref().unwrap()[j][d] };
                        let scr1 = cvj + tsc_v[yoffset];
                        if scr1 > vr {
                            vr = scr1;
                            shr = yoffset as i32;
                            rmode = MODE_J;
                        }
                        // scr2: R_alpha[y+yoffset][j][d] (R deck); self-loop -> in-progress vr. Rmode=1.
                        let cvr = if ch == v { vr } else { alpha.r[ch].as_ref().unwrap()[j][d] };
                        let scr2 = cvr + tsc_v[yoffset];
                        if scr2 > vr {
                            vr = scr2;
                            shr = yoffset as i32;
                            rmode = MODE_R;
                        }
                    }
                    let i = j - d + 1;
                    vj += sing_sc(esc_v, dsq[i]);
                    vl += sing_sc(esc_v, dsq[i]);
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[j][d] = vj;
                    dl[j][d] = vl;
                    dr[j][d] = vr;
                    if want_shadow {
                        sh_j[j][d] = shj;
                        sh_l[j][d] = shl;
                        sh_r[j][d] = shr;
                        sh_lm[j][d] = lmode;
                        sh_rm[j][d] = rmode;
                    }
                }
                if stt == ML_ST && allow_begin {
                    for d in 1..=(jp as usize) {
                        let vj = dj[j][d];
                        let vl = dl[j][d];
                        if !len_correx {
                            if vj > r_sc {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj;
                            }
                            if vl > r_sc {
                                r_mode = MODE_L;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vl;
                            }
                        } else {
                            let psc = p1 + (l - d as i32) as f32 * p2;
                            if vj + psc > r_sc {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj + psc;
                            }
                            if vl + psc > r_sc {
                                r_mode = MODE_L;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vl + psc;
                            }
                        }
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            // C truncyk.c:1866-1997
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                dj[j][0] = IMPOSSIBLE;
                dl[j][0] = IMPOSSIBLE;
                dr[j][0] = IMPOSSIBLE;
                for d in 1..=(jp as usize) {
                    let mut vj = endsc_v + elself * ((d - 1) as f32);
                    let mut vl = IMPOSSIBLE;
                    let mut vr = if d == 1 { 0.0 } else { IMPOSSIBLE };
                    let mut shj = USED_EL;
                    let mut shl = 0i32;
                    let mut shr = USED_EL;
                    let mut lmode = 0;
                    let mut rmode = MODE_J;
                    // First loop: J and R (depend on j-1,d-1)
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        let cvj = if ch == v { dj[j - 1][d - 1] } else { alpha.j[ch].as_ref().unwrap()[j - 1][d - 1] };
                        let scj = cvj + tsc_v[yoffset];
                        if scj > vj {
                            vj = scj;
                            shj = yoffset as i32;
                        }
                        if d > 1 {
                            let cvr = if ch == v { dr[j - 1][d - 1] } else { alpha.r[ch].as_ref().unwrap()[j - 1][d - 1] };
                            let scr = cvr + tsc_v[yoffset];
                            if scr > vr {
                                vr = scr;
                                shr = yoffset as i32;
                                rmode = MODE_R;
                            }
                        }
                    }
                    // Second loop: L (depends on fully-computed J cell alpha[v][j][d])
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        let cvj = if ch == v { vj } else { alpha.j[ch].as_ref().unwrap()[j][d] };
                        let scl1 = cvj + tsc_v[yoffset];
                        if scl1 > vl {
                            vl = scl1;
                            shl = yoffset as i32;
                            lmode = MODE_J;
                        }
                        let cvl_read = if ch == v { vl } else { alpha.l[ch].as_ref().unwrap()[j][d] };
                        let scl2 = cvl_read + tsc_v[yoffset];
                        if scl2 > vl {
                            vl = scl2;
                            shl = yoffset as i32;
                            lmode = MODE_L;
                        }
                    }
                    vj += sing_sc(esc_v, dsq[j]);
                    vr += sing_sc(esc_v, dsq[j]);
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[j][d] = vj;
                    dl[j][d] = vl;
                    dr[j][d] = vr;
                    if want_shadow {
                        sh_j[j][d] = shj;
                        sh_l[j][d] = shl;
                        sh_r[j][d] = shr;
                        sh_lm[j][d] = lmode;
                        sh_rm[j][d] = rmode;
                    }
                }
                if stt == MR_ST && allow_begin {
                    for d in 1..=(jp as usize) {
                        let vj = dj[j][d];
                        let vr = dr[j][d];
                        if !len_correx {
                            if vj > r_sc {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj;
                            }
                            if vr > r_sc {
                                r_mode = MODE_R;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vr;
                            }
                        } else {
                            let psc = p1 + (l - d as i32) as f32 * p2;
                            if vj + psc > r_sc {
                                r_mode = MODE_J;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vj + psc;
                            }
                            if vr + psc > r_sc {
                                r_mode = MODE_R;
                                r_v = v as i32;
                                r_j = j as i32;
                                r_i = j as i32 - d as i32 + 1;
                                r_sc = vr + psc;
                            }
                        }
                    }
                }
            }
        } else {
            panic!("tr_inside: inconceivable state type {stt}");
        }

        // Commit decks + shadows for v.
        alpha.j[v] = Some(dj);
        alpha.l[v] = Some(dl);
        alpha.r[v] = Some(dr);
        if stt == B_ST {
            alpha.t[v] = Some(dt);
        }
        if want_shadow {
            if let Some(sh) = shadow.as_deref_mut() {
                sh.j[v] = Some(sh_j);
                sh.l[v] = Some(sh_l);
                sh.r[v] = Some(sh_r);
                if stt == B_ST {
                    sh.t[v] = Some(sh_t);
                }
                sh.lmode[v] = Some(sh_lm);
                sh.rmode[v] = Some(sh_rm);
            }
        }

        // vroot best-hit check (C truncyk.c:2041-2056)
        if v == vroot {
            let vj = alpha.j[v].as_ref().unwrap()[j0 as usize][w as usize];
            let vl = alpha.l[v].as_ref().unwrap()[j0 as usize][w as usize];
            let vr = alpha.r[v].as_ref().unwrap()[j0 as usize][w as usize];
            if !len_correx {
                if vj > r_sc {
                    r_mode = MODE_J;
                    r_v = v as i32;
                    r_j = j0;
                    r_i = j0 - w + 1;
                    r_sc = vj;
                }
                if vl > r_sc {
                    r_mode = MODE_L;
                    r_v = v as i32;
                    r_j = j0;
                    r_i = j0 - w + 1;
                    r_sc = vl;
                }
                if vr > r_sc {
                    r_mode = MODE_R;
                    r_v = v as i32;
                    r_j = j0;
                    r_i = j0 - w + 1;
                    r_sc = vr;
                }
            } else {
                let psc = p1 + (l - w) as f32 * p2;
                if vj + psc > r_sc {
                    r_mode = MODE_J;
                    r_v = v as i32;
                    r_j = j0;
                    r_i = j0 - w + 1;
                    r_sc = vj + psc;
                }
                if vl + psc > r_sc {
                    r_mode = MODE_L;
                    r_v = v as i32;
                    r_j = j0;
                    r_i = j0 - w + 1;
                    r_sc = vl + psc;
                }
                if vr + psc > r_sc {
                    r_mode = MODE_R;
                    r_v = v as i32;
                    r_j = j0;
                    r_i = j0 - w + 1;
                    r_sc = vr + psc;
                }
            }
        }

        // v==0: write r_sc into the 0-deck corner + LOCAL_BEGIN shadow (C:2058-2068)
        if v == 0 {
            let jc = j0 as usize;
            let dc = w as usize;
            alpha.j[0].as_mut().unwrap()[jc][dc] = r_sc;
            alpha.l[0].as_mut().unwrap()[jc][dc] = r_sc;
            alpha.r[0].as_mut().unwrap()[jc][dc] = r_sc;
            if want_shadow {
                if let Some(sh) = shadow.as_deref_mut() {
                    sh.j[0].as_mut().unwrap()[jc][dc] = USED_LOCAL_BEGIN;
                    sh.l[0].as_mut().unwrap()[jc][dc] = USED_LOCAL_BEGIN;
                    sh.r[0].as_mut().unwrap()[jc][dc] = USED_LOCAL_BEGIN;
                    sh.lmode[0].as_mut().unwrap()[jc][dc] = r_mode;
                    sh.rmode[0].as_mut().unwrap()[jc][dc] = r_mode;
                }
            }
        }
    }

    let _ = m;
    TrInsideRet {
        sc: r_sc,
        mode: r_mode,
        v: r_v,
        i: r_i,
        j: r_j,
    }
}

/// The J/L/R beta deck-set (C `BetaMats_t`). NOTE the asymmetry, faithful to C:
/// `j` is a full vjd deck per state (indexed `[v][j][d]`, plus an EL deck at `M`),
/// but `l`/`r` are 1-D rows indexed `[v][j]` only (no `d`). No T (outside is only
/// run on unbifurcated subgraphs).
pub(crate) struct BetaMats {
    pub j: Vec<Option<VjdDeck>>, // 0..=M (M = EL deck)
    pub l: Vec<Vec<f32>>,        // 0..=M, each row length L+2, indexed by j
    pub r: Vec<Vec<f32>>,        // 0..=M, each row length L+2, indexed by j
}
impl BetaMats {
    fn new(m: usize, l: i32) -> Self {
        BetaMats {
            j: (0..=m).map(|_| None).collect(),
            l: (0..=m).map(|_| vec![IMPOSSIBLE; (l + 2) as usize]).collect(),
            r: (0..=m).map(|_| vec![IMPOSSIBLE; (l + 2) as usize]).collect(),
        }
    }
}

/// `esl_abc_FAvgScore`-style singlet with the emission read from `esc`: canonical
/// `esc[di]`, degenerate = uniform mean (matches `cm_dpsmall::sing_sc`).
#[inline]
fn favg(esc: &[f32], di: u8) -> f32 {
    sing_sc(esc, di)
}

// ================================================================
// tr_outside() — C truncyk.c:2184-2717
// Truncated CYK Outside over an unbifurcated segment vroot..vend. Fills beta J
// (vjd, + EL at M) and beta L/R (1-D per state). Returns best local hit
// (b_sc, b_mode, b_v, b_j) from the L/R marginal mini-recursions.
// `beta` is filled fresh (C arg_beta==NULL path).
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_outside(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    vroot: usize,
    vend: usize,
    i0: i32,
    j0: i32,
    _do_full: bool,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    beta: &mut BetaMats,
) -> (f32, i32, i32, i32) {
    let _ = r_allow_j;
    let big_m = cm.m as usize;
    let w = j0 - i0 + 1;

    // Initialize the root deck + its split set (C:2237-2287).
    let w1 = cm.nodemap[cm.ndidx[vroot] as usize] as usize;
    let w2: usize = if cm.sttype[vroot] as i32 == B_ST {
        w1
    } else {
        (cm.cfirst[w1] - 1) as usize
    };

    for v in w1..=w2 {
        let stt = cm.sttype[v] as i32;
        let mut allow_begin = vroot == 0;
        if stt == IL_ST || stt == IR_ST || stt == S_ST || stt == D_ST || stt == E_ST {
            allow_begin = false;
        }
        let mut dj = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        for jp in 0..=w {
            let j = (i0 + jp - 1) as usize;
            for d in 0..=(jp as usize) {
                dj[j][d] = if allow_begin { 0.0 } else { IMPOSSIBLE };
            }
            beta.l[v][j] = if allow_begin { 0.0 } else { IMPOSSIBLE };
            beta.r[v][j] = if allow_begin { 0.0 } else { IMPOSSIBLE };
        }
        beta.l[v][(i0 + w) as usize] = if allow_begin { 0.0 } else { IMPOSSIBLE };
        beta.r[v][(i0 + w) as usize] = if allow_begin { 0.0 } else { IMPOSSIBLE };
        beta.j[v] = Some(dj);
    }
    beta.j[vroot].as_mut().unwrap()[j0 as usize][w as usize] = 0.0;
    beta.l[vroot][i0 as usize] = if r_allow_l { 0.0 } else { IMPOSSIBLE };
    beta.r[vroot][j0 as usize] = if r_allow_r { 0.0 } else { IMPOSSIBLE };

    // Initialize EL deck (C:2292-2302).
    let mut el = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
    for jp in 0..=w {
        let j = (i0 + jp - 1) as usize;
        for d in 0..=(jp as usize) {
            el[j][d] = IMPOSSIBLE;
        }
        beta.l[big_m][j] = IMPOSSIBLE;
        beta.r[big_m][j] = IMPOSSIBLE;
    }
    beta.j[big_m] = Some(el);

    // vroot -> EL (C:2304-2350). Marginal L/R don't transition to EL.
    if not_impossible(cm.endsc[vroot]) {
        let elself = cm.el_selfsc;
        let endsc_r = cm.endsc[vroot];
        let esc_r = &cm.esc[vroot];
        let elm = beta.j[big_m].as_mut().unwrap();
        match cm.sttype[vroot] as i32 {
            MP_ST => {
                if w >= 2 {
                    let esc = pair_sc(esc_r, dsq[i0 as usize], dsq[j0 as usize]);
                    let jj = (j0 - 1) as usize;
                    let dd = (w - 2) as usize;
                    elm[jj][dd] = endsc_r + elself * (w - 2) as f32 + esc;
                    if elm[jj][dd] < IMPOSSIBLE {
                        elm[jj][dd] = IMPOSSIBLE;
                    }
                }
            }
            ML_ST | IL_ST => {
                if w >= 1 {
                    let esc = sing_sc(esc_r, dsq[i0 as usize]);
                    let jj = j0 as usize;
                    let dd = (w - 1) as usize;
                    elm[jj][dd] = endsc_r + elself * (w - 1) as f32 + esc;
                    if elm[jj][dd] < IMPOSSIBLE {
                        elm[jj][dd] = IMPOSSIBLE;
                    }
                }
            }
            MR_ST | IR_ST => {
                if w >= 1 {
                    let esc = sing_sc(esc_r, dsq[j0 as usize]);
                    let jj = (j0 - 1) as usize;
                    let dd = (w - 1) as usize;
                    elm[jj][dd] = endsc_r + elself * (w - 1) as f32 + esc;
                    // C:2339 latent bug: clamp writes to [j0][W-1], NOT [j0-1][W-1].
                    if elm[jj][dd] < IMPOSSIBLE {
                        elm[j0 as usize][dd] = IMPOSSIBLE;
                    }
                }
            }
            S_ST | D_ST => {
                let jj = j0 as usize;
                let dd = w as usize;
                elm[jj][dd] = endsc_r + elself * w as f32;
                if elm[jj][dd] < IMPOSSIBLE {
                    elm[jj][dd] = IMPOSSIBLE;
                }
            }
            _ => panic!("tr_outside: bogus parent state at vroot"),
        }
    }

    let mut b_sc = IMPOSSIBLE;
    let mut b_v: i32 = -1;
    let mut b_j: i32 = -1;
    let mut b_mode: i32 = -1;

    // Main loop through decks v = w2+1 .. vend (C:2367-2681).
    for v in (w2 + 1)..=vend {
        let stt = cm.sttype[v] as i32;
        let mut allow_begin = vroot == 0;
        if stt == IL_ST || stt == IR_ST || stt == S_ST || stt == D_ST || stt == E_ST {
            allow_begin = false;
        }

        let mut dj = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        for jp in (0..=w).rev() {
            let j = (i0 + jp - 1) as usize;
            for d in (0..=(jp as usize)).rev() {
                dj[j][d] = if allow_begin { 0.0 } else { IMPOSSIBLE };
            }
            beta.l[v][j] = if allow_begin { 0.0 } else { IMPOSSIBLE };
            beta.r[v][j] = if allow_begin { 0.0 } else { IMPOSSIBLE };
        }
        beta.l[v][(i0 + w) as usize] = IMPOSSIBLE;

        let plast_v = cm.plast[v];
        let pnum_v = cm.pnum[v];
        let esc_v = &cm.esc[v];
        let lmv = &lm[v];
        let rmv = &rm[v];

        // beta.L mini-recursion (C:2405-2467).
        if r_allow_l {
            for j in (i0)..=(j0 + 1) {
                let mut y = plast_v;
                while y > plast_v - pnum_v {
                    if y < vroot as i32 {
                        y -= 1;
                        continue;
                    }
                    let yu = y as usize;
                    let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                    match cm.sttype[yu] as i32 {
                        MP_ST => {
                            if j > i0 {
                                let esc = lm[yu][dsq[(j - 1) as usize] as usize];
                                let sc = beta.l[yu][(j - 1) as usize] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.l[v][j as usize] {
                                    beta.l[v][j as usize] = sc;
                                }
                            }
                        }
                        ML_ST | IL_ST => {
                            if j > i0 {
                                // C:2427-2430: canonical reads esc[y]; degenerate reads
                                // esc[v] (latent quirk, unreachable on non-degenerate data).
                                let di = dsq[(j - 1) as usize];
                                let esc = if (di as usize) < K {
                                    cm.esc[yu][di as usize]
                                } else {
                                    favg(esc_v, di)
                                };
                                let sc = beta.l[yu][(j - 1) as usize] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.l[v][j as usize] {
                                    beta.l[v][j as usize] = sc;
                                }
                            }
                        }
                        MR_ST | IR_ST | S_ST | E_ST | D_ST => {
                            let sc = beta.l[yu][j as usize] + cm.tsc[yu][voffset];
                            if sc > beta.l[v][j as usize] {
                                beta.l[v][j as usize] = sc;
                            }
                        }
                        _ => panic!("tr_outside L: bogus parent type"),
                    }
                    y -= 1;
                }
                let mut esc = 0.0f32;
                if j <= j0 {
                    if stt == MP_ST {
                        esc = lmv[dsq[j as usize] as usize];
                    } else if stt == ML_ST || stt == IL_ST {
                        esc = sing_sc(esc_v, dsq[j as usize]);
                    }
                }
                if beta.l[v][j as usize] + esc > b_sc {
                    b_sc = beta.l[v][j as usize] + esc;
                    b_v = v as i32;
                    b_j = j;
                    b_mode = MODE_L;
                }
            }
        }

        // beta.R mini-recursion (C:2470-2532).
        if r_allow_r {
            let mut j = j0;
            while j >= i0 - 1 {
                let mut y = plast_v;
                while y > plast_v - pnum_v {
                    if y < vroot as i32 {
                        y -= 1;
                        continue;
                    }
                    let yu = y as usize;
                    let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                    match cm.sttype[yu] as i32 {
                        MP_ST => {
                            if j < j0 {
                                let esc = rm[yu][dsq[(j + 1) as usize] as usize];
                                let sc = beta.r[yu][(j + 1) as usize] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.r[v][j as usize] {
                                    beta.r[v][j as usize] = sc;
                                }
                            }
                        }
                        MR_ST | IR_ST => {
                            if j < j0 {
                                let esc = sing_sc(&cm.esc[yu], dsq[(j + 1) as usize]);
                                let sc = beta.r[yu][(j + 1) as usize] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.r[v][j as usize] {
                                    beta.r[v][j as usize] = sc;
                                }
                            }
                        }
                        ML_ST | IL_ST | S_ST | E_ST | D_ST => {
                            let sc = beta.r[yu][j as usize] + cm.tsc[yu][voffset];
                            if sc > beta.r[v][j as usize] {
                                beta.r[v][j as usize] = sc;
                            }
                        }
                        _ => panic!("tr_outside R: bogus parent type"),
                    }
                    y -= 1;
                }
                let mut esc = 0.0f32;
                if j >= i0 {
                    if stt == MP_ST {
                        esc = rmv[dsq[j as usize] as usize];
                    } else if stt == MR_ST || stt == IR_ST {
                        esc = sing_sc(esc_v, dsq[j as usize]);
                    }
                }
                if beta.r[v][j as usize] + esc > b_sc {
                    b_sc = beta.r[v][j as usize] + esc;
                    b_v = v as i32;
                    b_j = j;
                    b_mode = MODE_R;
                }
                j -= 1;
            }
        }

        // main J recursion (C:2534-2606).
        for jp in (0..=w).rev() {
            let j = i0 + jp - 1;
            for d in (0..=jp).rev() {
                let i = j - d + 1;
                let mut cur = dj[j as usize][d as usize];
                let mut y = plast_v;
                while y > plast_v - pnum_v {
                    if y < vroot as i32 {
                        y -= 1;
                        continue;
                    }
                    let yu = y as usize;
                    let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                    // Self-loop (IL/IR): v is its own parent; beta.j[v] not yet
                    // committed, so read the in-progress local `dj` deck (C reads the
                    // partially-filled beta->J[v], whose higher j/d cells are done).
                    let sl = yu == v;
                    macro_rules! bj { ($a:expr,$b:expr) => { (if sl { &dj } else { beta.j[yu].as_ref().unwrap() })[$a][$b] } }
                    match cm.sttype[yu] as i32 {
                        MP_ST => {
                            if j != j0 && d != jp {
                                let esc = pair_sc(&cm.esc[yu], dsq[(i - 1) as usize], dsq[(j + 1) as usize]);
                                let t = cm.tsc[yu][voffset];
                                let s1 = bj!((j + 1) as usize, (d + 2) as usize) + t + esc;
                                if s1 > cur {
                                    cur = s1;
                                }
                                let s2 = beta.l[yu][(i - 1) as usize] + t + esc;
                                if s2 > cur {
                                    cur = s2;
                                }
                                let s3 = beta.r[yu][(j + 1) as usize] + t + esc;
                                if s3 > cur {
                                    cur = s3;
                                }
                            }
                        }
                        ML_ST | IL_ST => {
                            if d != jp {
                                let esc = sing_sc(&cm.esc[yu], dsq[(i - 1) as usize]);
                                let t = cm.tsc[yu][voffset];
                                let s1 = bj!(j as usize, (d + 1) as usize) + t + esc;
                                if s1 > cur {
                                    cur = s1;
                                }
                                let s2 = beta.l[yu][(i - 1) as usize] + t + esc;
                                if s2 > cur {
                                    cur = s2;
                                }
                                let s3 = beta.r[yu][j as usize] + t + esc;
                                if s3 > cur {
                                    cur = s3;
                                }
                            }
                        }
                        MR_ST | IR_ST => {
                            if j != j0 {
                                let esc = sing_sc(&cm.esc[yu], dsq[(j + 1) as usize]);
                                let t = cm.tsc[yu][voffset];
                                let s1 = bj!((j + 1) as usize, (d + 1) as usize) + t + esc;
                                if s1 > cur {
                                    cur = s1;
                                }
                                let s2 = beta.l[yu][i as usize] + t + esc;
                                if s2 > cur {
                                    cur = s2;
                                }
                                let s3 = beta.r[yu][(j + 1) as usize] + t + esc;
                                if s3 > cur {
                                    cur = s3;
                                }
                            }
                        }
                        S_ST | E_ST | D_ST => {
                            let sc = bj!(j as usize, d as usize) + cm.tsc[yu][voffset];
                            if sc > cur {
                                cur = sc;
                            }
                        }
                        _ => panic!("tr_outside J: bogus parent type"),
                    }
                    y -= 1;
                }
                dj[j as usize][d as usize] = cur;
            }
        }

        // v -> EL transitions (beta J only) (C:2608-2670).
        if not_impossible(cm.endsc[v]) {
            let endsc_v = cm.endsc[v];
            let elself = cm.el_selfsc;
            for jp in 0..=w {
                let j = i0 - 1 + jp;
                for d in 0..=jp {
                    let i = j - d + 1;
                    let sc = match stt {
                        MP_ST => {
                            if j == j0 || d == jp {
                                continue;
                            }
                            let esc = pair_sc(esc_v, dsq[(i - 1) as usize], dsq[(j + 1) as usize]);
                            dj[(j + 1) as usize][(d + 2) as usize] + endsc_v + elself * d as f32 + esc
                        }
                        ML_ST | IL_ST => {
                            if d == jp {
                                continue;
                            }
                            let esc = sing_sc(esc_v, dsq[(i - 1) as usize]);
                            dj[j as usize][(d + 1) as usize] + endsc_v + elself * d as f32 + esc
                        }
                        MR_ST | IR_ST => {
                            if j == j0 {
                                continue;
                            }
                            let esc = sing_sc(esc_v, dsq[(j + 1) as usize]);
                            dj[(j + 1) as usize][(d + 1) as usize] + endsc_v + elself * d as f32 + esc
                        }
                        S_ST | D_ST | E_ST => dj[j as usize][d as usize] + endsc_v + elself * d as f32,
                        _ => panic!("tr_outside EL: bogus parent state"),
                    };
                    let elm = beta.j[big_m].as_mut().unwrap();
                    if sc > elm[j as usize][d as usize] {
                        elm[j as usize][d as usize] = sc;
                    }
                }
            }
        }

        beta.j[v] = Some(dj);
    }

    (b_sc, b_mode, b_v, b_j)
}

// ---- vji deck management (coords [v][jp][ip], jp 0..j0-j1, ip 0..i1-i0) ----
// Same underlying type as vjd decks, so they reuse AlphaMats/ShadowMats.
fn new_vji_deck(i0: i32, i1: i32, j1: i32, j0: i32, init: f32) -> VjdDeck {
    vec![vec![init; (i1 - i0 + 1) as usize]; (j0 - j1 + 1) as usize]
}
fn new_vji_shadow(i0: i32, i1: i32, j1: i32, j0: i32, init: i32) -> VjdShadow {
    vec![vec![init; (i1 - i0 + 1) as usize]; (j0 - j1 + 1) as usize]
}

// ================================================================
// tr_vinside() — C truncyk.c:2730-3370
// Inside-type truncated CYK for a V-problem (unbifurcated segment r..z aligned to
// the outer subseq i0..i1 / j1..j0), in vji coordinates. Fills alpha J/L/R and (if
// requested) shadow J/L/R/Lmode/Rmode. Returns best local hit (b_sc,mode,v,i,j).
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_vinside(
    cm: &CM,
    dsq: &[u8],
    _l: i32,
    r: usize,
    z: usize,
    i0: i32,
    i1: i32,
    j1: i32,
    j0: i32,
    use_el: bool,
    _do_full: bool,
    allow_begin: bool,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    z_allow_j: bool,
    z_allow_l: bool,
    z_allow_r: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    alpha: &mut AlphaMats,
    mut shadow: Option<&mut ShadowMats>,
) -> TrInsideRet {
    let want_shadow = shadow.is_some();
    let elself = cm.el_selfsc;
    let jspan = (j0 - j1) as usize; // max jp
    let ispan = (i1 - i0) as usize; // max ip

    let mut b_v: i32 = -1;
    let mut b_i: i32 = i0;
    let mut b_j: i32 = j0;
    let mut b_mode: i32 = MODE_J;
    let mut b_sc: f32 = IMPOSSIBLE;

    let w1 = cm.nodemap[cm.ndidx[z] as usize] as usize;
    let w2 = (cm.cfirst[w1] - 1) as usize;

    // Init split-set decks w1..w2 to IMPOSSIBLE (C:2791-2808).
    for v in w1..=w2 {
        alpha.j[v] = Some(new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE));
        alpha.l[v] = Some(new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE));
        alpha.r[v] = Some(new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE));
    }

    // Boundary condition at (jp=0, ip=i1-i0) (C:2833-2912).
    let ip0 = ispan;
    if !use_el {
        if z_allow_j {
            alpha.j[z].as_mut().unwrap()[0][ip0] = 0.0;
        }
        if z_allow_l {
            alpha.l[z].as_mut().unwrap()[0][ip0] = 0.0;
        }
        if z_allow_r {
            alpha.r[z].as_mut().unwrap()[0][ip0] = 0.0;
        }
    } else if z_allow_j {
        if want_shadow {
            if let Some(sh) = shadow.as_deref_mut() {
                sh.j[z] = Some(new_vji_shadow(i0, i1, j1, j0, 0));
                sh.l[z] = Some(new_vji_shadow(i0, i1, j1, j0, 0));
                sh.r[z] = Some(new_vji_shadow(i0, i1, j1, j0, 0));
                sh.lmode[z] = Some(new_vji_shadow(i0, i1, j1, j0, 0));
                sh.rmode[z] = Some(new_vji_shadow(i0, i1, j1, j0, 0));
            }
        }
        let endsc_z = cm.endsc[z];
        let esc_z = &cm.esc[z];
        let djz = alpha.j[z].as_mut().unwrap();
        match cm.sttype[z] as i32 {
            D_ST | S_ST => {
                djz[0][ip0] = endsc_z + elself * ((j1 - (i1) + 1) as f32);
                if want_shadow {
                    if let Some(sh) = shadow.as_deref_mut() {
                        sh.j[z].as_mut().unwrap()[0][ip0] = USED_EL;
                    }
                }
            }
            MP_ST => {
                if i0 != i1 && j1 != j0 {
                    let mut val = endsc_z + elself * ((j1 - i1 + 1) as f32);
                    val += pair_sc(esc_z, dsq[(i1 - 1) as usize], dsq[(j1 + 1) as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    djz[1][ip0 - 1] = val;
                    if want_shadow {
                        if let Some(sh) = shadow.as_deref_mut() {
                            sh.j[z].as_mut().unwrap()[1][ip0 - 1] = USED_EL;
                        }
                    }
                }
            }
            ML_ST | IL_ST => {
                if i0 != i1 {
                    let mut val = endsc_z + elself * ((j1 - i1 + 1) as f32);
                    val += sing_sc(esc_z, dsq[(i1 - 1) as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    djz[0][ip0 - 1] = val;
                    if want_shadow {
                        if let Some(sh) = shadow.as_deref_mut() {
                            sh.j[z].as_mut().unwrap()[0][ip0 - 1] = USED_EL;
                        }
                    }
                }
            }
            MR_ST | IR_ST => {
                if j1 != j0 {
                    let mut val = endsc_z + elself * ((j1 - i1 + 1) as f32);
                    val += sing_sc(esc_z, dsq[(j1 + 1) as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    djz[1][ip0] = val;
                    if want_shadow {
                        if let Some(sh) = shadow.as_deref_mut() {
                            sh.j[z].as_mut().unwrap()[1][ip0] = USED_EL;
                        }
                    }
                }
            }
            _ => {}
        }
        alpha.l[z].as_mut().unwrap()[0][ip0] = IMPOSSIBLE;
        alpha.r[z].as_mut().unwrap()[0][ip0] = IMPOSSIBLE;
        if want_shadow {
            if let Some(sh) = shadow.as_deref_mut() {
                sh.l[z].as_mut().unwrap()[0][ip0] = USED_EL;
                sh.r[z].as_mut().unwrap()[0][ip0] = USED_EL;
            }
        }
    } else {
        panic!("tr_vinside: bad useEL/z_allow_J combination");
    }

    // Empty-sequence special case (C:2914-2941).
    if r == 0 {
        b_v = z as i32;
        b_i = i1;
        b_j = j1;
        b_sc = IMPOSSIBLE;
        b_mode = MODE_T;
        let jz = alpha.j[z].as_ref().unwrap();
        let lz = alpha.l[z].as_ref().unwrap();
        let rz = alpha.r[z].as_ref().unwrap();
        if z_allow_j && jz[0][ispan] > b_sc {
            b_sc = jz[0][ispan];
            b_mode = MODE_J;
        }
        if z_allow_l && lz[0][ispan] > b_sc {
            b_sc = lz[0][ispan];
            b_mode = MODE_L;
        }
        if z_allow_r && rz[0][ispan] > b_sc {
            b_sc = rz[0][ispan];
            b_mode = MODE_R;
        }
        if z == 0 {
            panic!("tr_vinside: potentially unhandled case (z==0)");
        }
    }

    // Main recursion v = w1-1 .. r (descending).
    let mut v_i = w1 as i32 - 1;
    while v_i >= r as i32 {
        let v = v_i as usize;
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let endsc_v = cm.endsc[v];
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];
        let has_el = use_el && not_impossible(endsc_v) && z_allow_j;

        let mut dj = new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE);
        let mut dl = new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE);
        let mut dr = new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE);
        let (mut sh_j, mut sh_l, mut sh_r, mut sh_lm, mut sh_rm) = if want_shadow {
            (
                new_vji_shadow(i0, i1, j1, j0, USED_EL),
                new_vji_shadow(i0, i1, j1, j0, USED_EL),
                new_vji_shadow(i0, i1, j1, j0, USED_EL),
                new_vji_shadow(i0, i1, j1, j0, 0),
                new_vji_shadow(i0, i1, j1, j0, 0),
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        if stt == D_ST || stt == S_ST {
            for jp in 0..=jspan {
                for ip in (0..=ispan).rev() {
                    let elsc = elself * ((jp as i32 + j1) - (ip as i32 + i0) + 1) as f32;
                    let mut vj = dj[jp][ip];
                    let mut shj = USED_EL;
                    if has_el {
                        let sc = endsc_v + elsc;
                        if sc > vj {
                            vj = sc;
                            shj = USED_EL;
                        }
                    }
                    let mut vl = dl[jp][ip];
                    let mut vr = dr[jp][ip];
                    let mut shl = USED_EL;
                    let mut shr = USED_EL;
                    let mut lmode = 0;
                    let mut rmode = 0;
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        if z_allow_j {
                            let cv = if ch == v { vj } else { alpha.j[ch].as_ref().unwrap()[jp][ip] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vj {
                                vj = sc;
                                shj = yoffset as i32;
                            }
                        }
                        if r_allow_l {
                            let cv = if ch == v { vl } else { alpha.l[ch].as_ref().unwrap()[jp][ip] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vl {
                                vl = sc;
                                shl = yoffset as i32;
                                lmode = MODE_L;
                            }
                        }
                        if r_allow_r {
                            let cv = if ch == v { vr } else { alpha.r[ch].as_ref().unwrap()[jp][ip] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vr {
                                vr = sc;
                                shr = yoffset as i32;
                                rmode = MODE_R;
                            }
                        }
                    }
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[jp][ip] = vj;
                    dl[jp][ip] = vl;
                    dr[jp][ip] = vr;
                    if want_shadow {
                        sh_j[jp][ip] = shj;
                        sh_l[jp][ip] = shl;
                        sh_r[jp][ip] = shr;
                        sh_lm[jp][ip] = lmode;
                        sh_rm[jp][ip] = rmode;
                    }
                }
            }
        } else if stt == MP_ST {
            for jp in 0..=jspan {
                let j = jp as i32 + j1;
                for ip in (0..=ispan).rev() {
                    let i = ip as i32 + i0;
                    let mut vj = dj[jp][ip];
                    let mut vl = dl[jp][ip];
                    let mut vr = dr[jp][ip];
                    let mut shj = USED_EL;
                    let mut shl = USED_EL;
                    let mut shr = USED_EL;
                    let mut lmode = 0;
                    let mut rmode = 0;
                    if has_el && jp > 0 && ip < ispan {
                        let sc = endsc_v + elself * ((j - i + 1 - 2) as f32);
                        if sc > vj {
                            vj = sc;
                            shj = USED_EL;
                        }
                    }
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        if z_allow_j && jp > 0 && ip < ispan {
                            let cv = if ch == v { dj[jp - 1][ip + 1] } else { alpha.j[ch].as_ref().unwrap()[jp - 1][ip + 1] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vj {
                                vj = sc;
                                shj = yoffset as i32;
                            }
                        }
                        if r_allow_l && ip < ispan {
                            // L from J child (Lmode=3)
                            let cvj = if ch == v { vj } else { alpha.j[ch].as_ref().unwrap()[jp][ip + 1] };
                            let sc = cvj + tsc_v[yoffset];
                            if sc > vl {
                                vl = sc;
                                shl = yoffset as i32;
                                lmode = MODE_J;
                            }
                            // L from L child (Lmode=2)
                            let cvl = if ch == v { dl[jp][ip + 1] } else { alpha.l[ch].as_ref().unwrap()[jp][ip + 1] };
                            let sc = cvl + tsc_v[yoffset];
                            if sc > vl {
                                vl = sc;
                                shl = yoffset as i32;
                                lmode = MODE_L;
                            }
                        }
                        if r_allow_r && jp > 0 {
                            // R from J child (Rmode=3)
                            let cvj = if ch == v { dj[jp - 1][ip] } else { alpha.j[ch].as_ref().unwrap()[jp - 1][ip] };
                            let sc = cvj + tsc_v[yoffset];
                            if sc > vr {
                                vr = sc;
                                shr = yoffset as i32;
                                rmode = MODE_J;
                            }
                            // R from R child (Rmode=1)
                            let cvr = if ch == v { dr[jp - 1][ip] } else { alpha.r[ch].as_ref().unwrap()[jp - 1][ip] };
                            let sc = cvr + tsc_v[yoffset];
                            if sc > vr {
                                vr = sc;
                                shr = yoffset as i32;
                                rmode = MODE_R;
                            }
                        }
                    }
                    if jp > 0 && ip < ispan {
                        vj += pair_sc(esc_v, dsq[i as usize], dsq[j as usize]);
                    }
                    if ip < ispan {
                        vl += lm[v][dsq[i as usize] as usize];
                    }
                    if jp > 0 {
                        vr += rm[v][dsq[j as usize] as usize];
                    }
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[jp][ip] = vj;
                    dl[jp][ip] = vl;
                    dr[jp][ip] = vr;
                    if want_shadow {
                        sh_j[jp][ip] = shj;
                        sh_l[jp][ip] = shl;
                        sh_r[jp][ip] = shr;
                        sh_lm[jp][ip] = lmode;
                        sh_rm[jp][ip] = rmode;
                    }
                    if allow_begin {
                        if r_allow_j && vj > b_sc {
                            b_mode = MODE_J;
                            b_v = v as i32;
                            b_j = j1 + jp as i32;
                            b_i = i0 + ip as i32;
                            b_sc = vj;
                        }
                        if r_allow_l && vl > b_sc {
                            b_mode = MODE_L;
                            b_v = v as i32;
                            b_j = j1 + jp as i32;
                            b_i = i0 + ip as i32;
                            b_sc = vl;
                        }
                        if r_allow_r && vr > b_sc {
                            b_mode = MODE_R;
                            b_v = v as i32;
                            b_j = j1 + jp as i32;
                            b_i = i0 + ip as i32;
                            b_sc = vr;
                        }
                    }
                }
            }
        } else if stt == ML_ST || stt == IL_ST {
            for jp in 0..=jspan {
                for ip in (0..=ispan).rev() {
                    let i = i0 + ip as i32;
                    let j = j1 + jp as i32;
                    let mut vj = dj[jp][ip];
                    let mut vl = dl[jp][ip];
                    let mut vr = dr[jp][ip];
                    let mut shj = USED_EL;
                    let mut shl = USED_EL;
                    let mut shr = USED_EL;
                    let mut lmode = 0;
                    let mut rmode = 0;
                    if has_el && ip < ispan {
                        let sc = endsc_v + elself * ((j - i + 1 - 1) as f32);
                        if sc > vj {
                            vj = sc;
                            shj = USED_EL;
                        }
                    }
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        if z_allow_j && ip < ispan {
                            let cv = if ch == v { dj[jp][ip + 1] } else { alpha.j[ch].as_ref().unwrap()[jp][ip + 1] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vj {
                                vj = sc;
                                shj = yoffset as i32;
                            }
                        }
                        if r_allow_l && ip < ispan {
                            let cv = if ch == v { dl[jp][ip + 1] } else { alpha.l[ch].as_ref().unwrap()[jp][ip + 1] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vl {
                                vl = sc;
                                shl = yoffset as i32;
                                lmode = MODE_L;
                            }
                        }
                        if r_allow_r {
                            let cvj = if ch == v { vj } else { alpha.j[ch].as_ref().unwrap()[jp][ip] };
                            let sc = cvj + tsc_v[yoffset];
                            if sc > vr {
                                vr = sc;
                                shr = yoffset as i32;
                                rmode = MODE_J;
                            }
                            let cvr = if ch == v { vr } else { alpha.r[ch].as_ref().unwrap()[jp][ip] };
                            let sc = cvr + tsc_v[yoffset];
                            if sc > vr {
                                vr = sc;
                                shr = yoffset as i32;
                                rmode = MODE_R;
                            }
                        }
                    }
                    if ip < ispan {
                        let e = sing_sc(esc_v, dsq[i as usize]);
                        vj += e;
                        vl += e;
                    }
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[jp][ip] = vj;
                    dl[jp][ip] = vl;
                    dr[jp][ip] = vr;
                    if want_shadow {
                        sh_j[jp][ip] = shj;
                        sh_l[jp][ip] = shl;
                        sh_r[jp][ip] = shr;
                        sh_lm[jp][ip] = lmode;
                        sh_rm[jp][ip] = rmode;
                    }
                    if stt == ML_ST && allow_begin {
                        if r_allow_j && vj > b_sc {
                            b_mode = MODE_J;
                            b_v = v as i32;
                            b_j = j1 + jp as i32;
                            b_i = i0 + ip as i32;
                            b_sc = vj;
                        }
                        if r_allow_l && vl > b_sc {
                            b_mode = MODE_L;
                            b_v = v as i32;
                            b_j = j1 + jp as i32;
                            b_i = i0 + ip as i32;
                            b_sc = vl;
                        }
                    }
                }
            }
        } else if stt == MR_ST || stt == IR_ST {
            for jp in 0..=jspan {
                let j = j1 + jp as i32;
                for ip in (0..=ispan).rev() {
                    let i = i0 + ip as i32;
                    let mut vj = dj[jp][ip];
                    let mut vl = dl[jp][ip];
                    let mut vr = dr[jp][ip];
                    let mut shj = USED_EL;
                    let mut shl = USED_EL;
                    let mut shr = USED_EL;
                    let mut lmode = 0;
                    let mut rmode = 0;
                    if has_el && jp > 0 {
                        let sc = endsc_v + elself * ((j - i + 1 - 1) as f32);
                        if sc > vj {
                            vj = sc;
                            shj = USED_EL;
                        }
                    }
                    for yoffset in 0..cnum {
                        let ch = cfirst + yoffset;
                        if z_allow_j && jp > 0 {
                            let cv = if ch == v { dj[jp - 1][ip] } else { alpha.j[ch].as_ref().unwrap()[jp - 1][ip] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vj {
                                vj = sc;
                                shj = yoffset as i32;
                            }
                        }
                        if r_allow_l {
                            let cvj = if ch == v { vj } else { alpha.j[ch].as_ref().unwrap()[jp][ip] };
                            let sc = cvj + tsc_v[yoffset];
                            if sc > vl {
                                vl = sc;
                                shl = yoffset as i32;
                                lmode = MODE_J;
                            }
                            let cvl = if ch == v { vl } else { alpha.l[ch].as_ref().unwrap()[jp][ip] };
                            let sc = cvl + tsc_v[yoffset];
                            if sc > vl {
                                vl = sc;
                                shl = yoffset as i32;
                                lmode = MODE_L;
                            }
                        }
                        if r_allow_r && jp > 0 {
                            let cv = if ch == v { dr[jp - 1][ip] } else { alpha.r[ch].as_ref().unwrap()[jp - 1][ip] };
                            let sc = cv + tsc_v[yoffset];
                            if sc > vr {
                                vr = sc;
                                shr = yoffset as i32;
                                rmode = MODE_R;
                            }
                        }
                    }
                    if jp > 0 {
                        let e = sing_sc(esc_v, dsq[j as usize]);
                        vj += e;
                        vr += e;
                    }
                    if vj < IMPOSSIBLE {
                        vj = IMPOSSIBLE;
                    }
                    if vl < IMPOSSIBLE {
                        vl = IMPOSSIBLE;
                    }
                    if vr < IMPOSSIBLE {
                        vr = IMPOSSIBLE;
                    }
                    dj[jp][ip] = vj;
                    dl[jp][ip] = vl;
                    dr[jp][ip] = vr;
                    if want_shadow {
                        sh_j[jp][ip] = shj;
                        sh_l[jp][ip] = shl;
                        sh_r[jp][ip] = shr;
                        sh_lm[jp][ip] = lmode;
                        sh_rm[jp][ip] = rmode;
                    }
                    if stt == MR_ST && allow_begin {
                        if r_allow_j && vj > b_sc {
                            b_mode = MODE_J;
                            b_v = v as i32;
                            b_j = j1 + jp as i32;
                            b_i = i0 + ip as i32;
                            b_sc = vj;
                        }
                        if r_allow_r && vr > b_sc {
                            b_mode = MODE_R;
                            b_v = v as i32;
                            b_j = j1 + jp as i32;
                            b_i = i0 + ip as i32;
                            b_sc = vr;
                        }
                    }
                }
            }
        } else {
            panic!("tr_vinside: non-V problem state type {stt}");
        }

        // v==r best-hit (C:3261-3269).
        if v == r {
            let vj = dj[jspan][0];
            let vl = dl[jspan][0];
            let vr = dr[jspan][0];
            if r_allow_j && vj > b_sc {
                b_mode = MODE_J;
                b_v = v as i32;
                b_j = j0;
                b_i = i0;
                b_sc = vj;
            }
            if r_allow_l && vl > b_sc {
                b_mode = MODE_L;
                b_v = v as i32;
                b_j = j0;
                b_i = i0;
                b_sc = vl;
            }
            if r_allow_r && vr > b_sc {
                b_mode = MODE_R;
                b_v = v as i32;
                b_j = j0;
                b_i = i0;
                b_sc = vr;
            }
        }

        // v==0: write best score into corner + LOCAL_BEGIN shadow (C:3272-3285).
        if v == 0 {
            dj[jspan][0] = b_sc;
            dl[jspan][0] = b_sc;
            dr[jspan][0] = b_sc;
            if want_shadow {
                sh_j[jspan][0] = USED_LOCAL_BEGIN;
                sh_l[jspan][0] = USED_LOCAL_BEGIN;
                sh_r[jspan][0] = USED_LOCAL_BEGIN;
                sh_lm[jspan][0] = b_mode;
                sh_rm[jspan][0] = b_mode;
            }
        }

        alpha.j[v] = Some(dj);
        alpha.l[v] = Some(dl);
        alpha.r[v] = Some(dr);
        if want_shadow {
            if let Some(sh) = shadow.as_deref_mut() {
                sh.j[v] = Some(sh_j);
                sh.l[v] = Some(sh_l);
                sh.r[v] = Some(sh_r);
                sh.lmode[v] = Some(sh_lm);
                sh.rmode[v] = Some(sh_rm);
            }
        }

        v_i -= 1;
    }

    TrInsideRet {
        sc: b_sc,
        mode: b_mode,
        v: b_v,
        i: b_i,
        j: b_j,
    }
}

// ================================================================
// tr_voutside() — C truncyk.c:3384-3838
// Outside-type truncated CYK for a V-problem (vji coords). Fills beta J (vji, + EL
// at M), beta L (1-D by ip) and beta R (1-D by jp). Returns nothing (fills `beta`).
// `beta` is filled fresh (C arg_beta==NULL path).
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_voutside(
    cm: &CM,
    dsq: &[u8],
    _l: i32,
    r: usize,
    z: usize,
    i0: i32,
    i1: i32,
    j1: i32,
    j0: i32,
    use_el: bool,
    _do_full: bool,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    z_allow_j: bool,
    _z_allow_l: bool,
    _z_allow_r: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    beta: &mut BetaMats,
) {
    let big_m = cm.m as usize;
    let jspan = (j0 - j1) as usize;
    let ispan = (i1 - i0) as usize;
    let elself = cm.el_selfsc;

    // Initialize root deck (C:3434-3460). No split-set init (V-problem).
    let mut djr = new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE);
    for jp in 0..=jspan {
        for ip in 0..=ispan {
            djr[jp][ip] = if r == 0 && r_allow_j { 0.0 } else { IMPOSSIBLE };
        }
        beta.r[r][jp] = if r == 0 && r_allow_r { 0.0 } else { IMPOSSIBLE };
    }
    for ip in 0..=ispan {
        beta.l[r][ip] = if r == 0 && r_allow_l { 0.0 } else { IMPOSSIBLE };
    }
    if r_allow_j {
        djr[jspan][0] = 0.0;
    }
    if r_allow_l {
        beta.l[r][0] = 0.0;
    }
    if r_allow_r {
        beta.r[r][jspan] = 0.0;
    }
    beta.j[r] = Some(djr);

    // EL deck init + vroot->EL (C:3462-3511). Marginal modes don't use EL.
    if use_el {
        beta.j[big_m] = Some(new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE));
    }
    if use_el && not_impossible(cm.endsc[r]) {
        let endsc_r = cm.endsc[r];
        let esc_r = &cm.esc[r];
        let elm = beta.j[big_m].as_mut().unwrap();
        match cm.sttype[r] as i32 {
            MP_ST => {
                if i0 != i1 && j1 != j0 {
                    let esc = pair_sc(esc_r, dsq[i0 as usize], dsq[j0 as usize]);
                    let mut val = endsc_r + elself * ((j0 - 1) - (i0 + 1) + 1) as f32 + esc;
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    elm[jspan - 1][1] = val;
                }
            }
            ML_ST | IL_ST => {
                if i0 != i1 {
                    let esc = sing_sc(esc_r, dsq[i0 as usize]);
                    let mut val = endsc_r + elself * (j0 - (i0 + 1) + 1) as f32 + esc;
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    elm[jspan][1] = val;
                }
            }
            MR_ST | IR_ST => {
                if j1 != j0 {
                    let esc = sing_sc(esc_r, dsq[j0 as usize]);
                    let mut val = endsc_r + elself * ((j0 - 1) - i0 + 1) as f32 + esc;
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    elm[jspan - 1][0] = val;
                }
            }
            S_ST | D_ST => {
                elm[jspan][0] = endsc_r + elself * (j0 - i0 + 1) as f32;
            }
            _ => panic!("tr_voutside: bogus parent state at r"),
        }
    }

    // Main loop v = r+1 .. z (C:3523-3801).
    for v in (r + 1)..=z {
        let stt = cm.sttype[v] as i32;
        let mut allow_begin = r == 0;
        if stt == IL_ST || stt == IR_ST || stt == S_ST || stt == D_ST || stt == E_ST {
            allow_begin = false;
        }

        let mut dj = new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE);
        for jp in (0..=jspan).rev() {
            for ip in 0..=ispan {
                dj[jp][ip] = if allow_begin && r_allow_j { 0.0 } else { IMPOSSIBLE };
            }
            beta.r[v][jp] = if allow_begin && r_allow_r { 0.0 } else { IMPOSSIBLE };
        }
        for ip in 0..=ispan {
            beta.l[v][ip] = if allow_begin && r_allow_l { 0.0 } else { IMPOSSIBLE };
        }

        let plast_v = cm.plast[v];
        let pnum_v = cm.pnum[v];

        // beta.L mini-recursion (C:3559-3604).
        if r_allow_l {
            for ip in 0..=ispan {
                let i = i0 + ip as i32;
                let mut y = plast_v;
                while y > plast_v - pnum_v {
                    if y < r as i32 {
                        y -= 1;
                        continue;
                    }
                    let yu = y as usize;
                    let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                    match cm.sttype[yu] as i32 {
                        MP_ST => {
                            if ip > 0 {
                                let esc = lm[yu][dsq[(i - 1) as usize] as usize];
                                let sc = beta.l[yu][ip - 1] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.l[v][ip] {
                                    beta.l[v][ip] = sc;
                                }
                            }
                        }
                        ML_ST | IL_ST => {
                            if ip > 0 {
                                let esc = sing_sc(&cm.esc[yu], dsq[(i - 1) as usize]);
                                let sc = beta.l[yu][ip - 1] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.l[v][ip] {
                                    beta.l[v][ip] = sc;
                                }
                            }
                        }
                        MR_ST | IR_ST | S_ST | E_ST | D_ST => {
                            let sc = beta.l[yu][ip] + cm.tsc[yu][voffset];
                            if sc > beta.l[v][ip] {
                                beta.l[v][ip] = sc;
                            }
                        }
                        _ => panic!("tr_voutside L: bogus parent type"),
                    }
                    y -= 1;
                }
                if beta.l[v][ip] < IMPOSSIBLE {
                    beta.l[v][ip] = IMPOSSIBLE;
                }
            }
        }

        // beta.R mini-recursion (C:3607-3652).
        if r_allow_r {
            for jp in (0..=jspan).rev() {
                let j = j1 + jp as i32;
                let mut y = plast_v;
                while y > plast_v - pnum_v {
                    if y < r as i32 {
                        y -= 1;
                        continue;
                    }
                    let yu = y as usize;
                    let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                    match cm.sttype[yu] as i32 {
                        MP_ST => {
                            if jp < jspan {
                                let esc = rm[yu][dsq[(j + 1) as usize] as usize];
                                let sc = beta.r[yu][jp + 1] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.r[v][jp] {
                                    beta.r[v][jp] = sc;
                                }
                            }
                        }
                        MR_ST | IR_ST => {
                            if jp < jspan {
                                let esc = sing_sc(&cm.esc[yu], dsq[(j + 1) as usize]);
                                let sc = beta.r[yu][jp + 1] + cm.tsc[yu][voffset] + esc;
                                if sc > beta.r[v][jp] {
                                    beta.r[v][jp] = sc;
                                }
                            }
                        }
                        ML_ST | IL_ST | S_ST | E_ST | D_ST => {
                            let sc = beta.r[yu][jp] + cm.tsc[yu][voffset];
                            if sc > beta.r[v][jp] {
                                beta.r[v][jp] = sc;
                            }
                        }
                        _ => panic!("tr_voutside R: bogus parent type"),
                    }
                    y -= 1;
                }
                if beta.r[v][jp] < IMPOSSIBLE {
                    beta.r[v][jp] = IMPOSSIBLE;
                }
            }
        }

        // main J recursion (C:3655-3729).
        if z_allow_j {
            for jp in (0..=jspan).rev() {
                let j = j1 + jp as i32;
                for ip in 0..=ispan {
                    let i = i0 + ip as i32;
                    let mut cur = dj[jp][ip];
                    let mut y = plast_v;
                    while y > plast_v - pnum_v {
                        if y < r as i32 {
                            y -= 1;
                            continue;
                        }
                        let yu = y as usize;
                        let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                        let t = cm.tsc[yu][voffset];
                        // Self-loop (IL/IR): read in-progress local `dj` (see tr_outside).
                        let sl = yu == v;
                        macro_rules! bj { ($a:expr,$b:expr) => { (if sl { &dj } else { beta.j[yu].as_ref().unwrap() })[$a][$b] } }
                        match cm.sttype[yu] as i32 {
                            MP_ST => {
                                if j != j0 && i != i0 {
                                    let esc = pair_sc(&cm.esc[yu], dsq[(i - 1) as usize], dsq[(j + 1) as usize]);
                                    let s1 = bj!(jp + 1, ip - 1) + t + esc;
                                    if s1 > cur {
                                        cur = s1;
                                    }
                                    let s2 = beta.l[yu][ip - 1] + t + esc;
                                    if s2 > cur {
                                        cur = s2;
                                    }
                                    let s3 = beta.r[yu][jp + 1] + t + esc;
                                    if s3 > cur {
                                        cur = s3;
                                    }
                                }
                            }
                            ML_ST | IL_ST => {
                                if i != i0 {
                                    let esc = sing_sc(&cm.esc[yu], dsq[(i - 1) as usize]);
                                    let s1 = bj!(jp, ip - 1) + t + esc;
                                    if s1 > cur {
                                        cur = s1;
                                    }
                                    let s2 = beta.l[yu][ip - 1] + t + esc;
                                    if s2 > cur {
                                        cur = s2;
                                    }
                                    let s3 = beta.r[yu][jp] + t + esc;
                                    if s3 > cur {
                                        cur = s3;
                                    }
                                }
                            }
                            MR_ST | IR_ST => {
                                if j != j0 {
                                    let esc = sing_sc(&cm.esc[yu], dsq[(j + 1) as usize]);
                                    let s1 = bj!(jp + 1, ip) + t + esc;
                                    if s1 > cur {
                                        cur = s1;
                                    }
                                    let s2 = beta.l[yu][ip] + t + esc;
                                    if s2 > cur {
                                        cur = s2;
                                    }
                                    let s3 = beta.r[yu][jp + 1] + t + esc;
                                    if s3 > cur {
                                        cur = s3;
                                    }
                                }
                            }
                            S_ST | E_ST | D_ST => {
                                let sc = bj!(jp, ip) + t;
                                if sc > cur {
                                    cur = sc;
                                }
                            }
                            _ => panic!("tr_voutside J: bogus parent type"),
                        }
                        y -= 1;
                    }
                    if cur < IMPOSSIBLE {
                        cur = IMPOSSIBLE;
                    }
                    dj[jp][ip] = cur;
                }
            }
        }

        // v->EL transitions (beta J only) (C:3732-3790). Dead in global mode
        // (endsc IMPOSSIBLE). NB C's S/D/E case has a stale-esc + missing-break
        // fall-through bug; unreachable here, so we compute the S/D/E value with
        // esc omitted and do not panic.
        if use_el && not_impossible(cm.endsc[v]) {
            let endsc_v = cm.endsc[v];
            let esc_v = &cm.esc[v];
            for jp in (0..=jspan).rev() {
                let j = j1 + jp as i32;
                for ip in 0..=ispan {
                    let i = i0 + ip as i32;
                    let cand = match stt {
                        MP_ST => {
                            if j != j0 && i != i0 {
                                let esc = pair_sc(esc_v, dsq[(i - 1) as usize], dsq[(j + 1) as usize]);
                                Some(dj[jp + 1][ip - 1] + endsc_v + elself * (j - i + 1) as f32 + esc)
                            } else {
                                None
                            }
                        }
                        ML_ST | IL_ST => {
                            if i != i0 {
                                let esc = sing_sc(esc_v, dsq[(i - 1) as usize]);
                                Some(dj[jp][ip - 1] + endsc_v + elself * (j - i + 1) as f32 + esc)
                            } else {
                                None
                            }
                        }
                        MR_ST | IR_ST => {
                            if j != j0 {
                                let esc = sing_sc(esc_v, dsq[(j + 1) as usize]);
                                Some(dj[jp + 1][ip] + endsc_v + elself * (j - i + 1) as f32 + esc)
                            } else {
                                None
                            }
                        }
                        S_ST | E_ST | D_ST => Some(dj[jp][ip] + endsc_v + elself * j as f32),
                        _ => None,
                    };
                    if let Some(sc) = cand {
                        let elm = beta.j[big_m].as_mut().unwrap();
                        if sc > elm[jp][ip] {
                            elm[jp][ip] = sc;
                        }
                        if elm[jp][ip] < IMPOSSIBLE {
                            elm[jp][ip] = IMPOSSIBLE;
                        }
                    }
                }
            }
        }

        beta.j[v] = Some(dj);
    }
}

/// C `InsertTraceNodewithMode`: append a node (parent `parent`, left/right child)
/// carrying marginal `mode`, link the parent's nxtl/nxtr, return the new index.
fn insert_trace_node_mode(
    tr: &mut Parsetree,
    parent: i32,
    is_left: bool,
    emitl: i32,
    emitr: i32,
    state: i32,
    mode: i8,
) -> i32 {
    let idx = tr.add_node_mode(emitl, emitr, state, -1, -1, parent, mode);
    if parent >= 0 {
        if is_left {
            tr.nxtl[parent as usize] = idx;
        } else {
            tr.nxtr[parent as usize] = idx;
        }
    }
    idx
}

/// C `CMSubtreeFindEnd` (cm.c): the END_E state closing the subtree rooted at `r`.
fn subtree_find_end(cm: &CM, r: usize) -> usize {
    let mut unsatisfied: i32 = 1;
    let mut r = r;
    while unsatisfied != 0 {
        if cm.sttype[r] as i32 == B_ST {
            unsatisfied += 1;
        }
        if cm.sttype[r] as i32 == E_ST {
            unsatisfied -= 1;
        }
        r += 1;
    }
    r - 1
}

const EL_ST: i32 = 8; // C EL_st (constants); used only for the E/EL traceback test

// ================================================================
// tr_insideT() — C truncyk.c:3858-4070
// Fill (with shadow) via tr_inside, then trace back the optimal parse into `tr`.
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_insidet(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    j0: i32,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    len_correx: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
) -> f32 {
    let m = cm.m as usize;
    let mut alpha = AlphaMats::new(m);
    let mut shadow = ShadowMats::new(m);
    let ret = tr_inside(
        cm, dsq, l, r, z, i0, j0, false, r == 0, r_allow_j, r_allow_l, r_allow_r, len_correx, lm,
        rm, &mut alpha, Some(&mut shadow),
    );
    let sc = ret.sc;
    let mut mode = ret.mode;
    let mut v = ret.v as usize;
    let mut i = ret.i;
    let mut j = ret.j;
    let mut d = j - i + 1;

    if r == 0 {
        insert_trace_node_mode(tr, tr.n - 1, true, i, j, v as i32, mode as i8);
    }

    // stack element: (j, k, mode, bifparent)
    let mut pda: Vec<(i32, i32, i32, i32)> = Vec::new();

    loop {
        let stt = cm.sttype[v] as i32;
        if stt == B_ST {
            let ju = j as usize;
            let du = d as usize;
            let k;
            if mode == MODE_J {
                k = shadow.j[v].as_ref().unwrap()[ju][du];
                pda.push((j, k, mode, tr.n - 1));
            } else if mode == MODE_L {
                k = shadow.l[v].as_ref().unwrap()[ju][du];
                pda.push((j, k, mode, tr.n - 1));
                mode = shadow.lmode[v].as_ref().unwrap()[ju][du];
            } else if mode == MODE_R {
                k = shadow.r[v].as_ref().unwrap()[ju][du];
                mode = shadow.rmode[v].as_ref().unwrap()[ju][du];
                pda.push((j, k, mode, tr.n - 1));
                mode = MODE_R;
            } else if mode == MODE_T {
                k = shadow.t[v].as_ref().unwrap()[ju][du];
                mode = MODE_L;
                pda.push((j, k, mode, tr.n - 1));
                mode = MODE_R;
            } else {
                panic!("tr_insidet: unknown mode {mode}");
            }
            j -= k;
            d -= k;
            i = j - d + 1;
            v = cm.cfirst[v] as usize;
            insert_trace_node_mode(tr, tr.n - 1, true, i, j, v as i32, mode as i8);
        } else if stt == E_ST || stt == EL_ST {
            let Some((pj, pk, pmode, bifparent)) = pda.pop() else {
                break;
            };
            let _ = pk;
            mode = pmode;
            d = pk;
            j = pj;
            let vp = tr.state[bifparent as usize] as usize;
            let y = cm.cnum[vp] as usize; // right child state
            i = j - d + 1;
            v = y;
            insert_trace_node_mode(tr, bifparent, false, i, j, v as i32, mode as i8);
        } else {
            let ju = j as usize;
            let du = d as usize;
            let (yoffset, nxtmode) = if mode == MODE_J {
                (shadow.j[v].as_ref().unwrap()[ju][du], MODE_J)
            } else if mode == MODE_L {
                (
                    shadow.l[v].as_ref().unwrap()[ju][du],
                    shadow.lmode[v].as_ref().unwrap()[ju][du],
                )
            } else if mode == MODE_R {
                (
                    shadow.r[v].as_ref().unwrap()[ju][du],
                    shadow.rmode[v].as_ref().unwrap()[ju][du],
                )
            } else {
                panic!("tr_insidet: unknown mode {mode}");
            };
            match stt {
                D_ST | S_ST => {}
                MP_ST => {
                    if mode == MODE_J {
                        i += 1;
                    }
                    if mode == MODE_L && d > 0 {
                        i += 1;
                    }
                    if mode == MODE_J {
                        j -= 1;
                    }
                    if mode == MODE_R && d > 0 {
                        j -= 1;
                    }
                }
                ML_ST => {
                    if mode == MODE_J {
                        i += 1;
                    }
                    if mode == MODE_L && d > 0 {
                        i += 1;
                    }
                }
                MR_ST => {
                    if mode == MODE_J {
                        j -= 1;
                    }
                    if mode == MODE_R && d > 0 {
                        j -= 1;
                    }
                }
                IL_ST => {
                    if mode == MODE_J {
                        i += 1;
                    }
                    if mode == MODE_L && d > 0 {
                        i += 1;
                    }
                }
                IR_ST => {
                    if mode == MODE_J {
                        j -= 1;
                    }
                    if mode == MODE_R && d > 0 {
                        j -= 1;
                    }
                }
                _ => panic!("tr_insidet: inconceivable state type {stt}"),
            }
            d = j - i + 1;

            if yoffset == USED_EL {
                v = m;
                if mode == MODE_J {
                    insert_trace_node_mode(tr, tr.n - 1, true, i, j, v as i32, mode as i8);
                }
            } else if yoffset == USED_LOCAL_BEGIN {
                panic!("tr_insidet: impossible local begin in traceback");
            } else {
                mode = nxtmode;
                v = (cm.cfirst[v] + yoffset) as usize;
                insert_trace_node_mode(tr, tr.n - 1, true, i, j, v as i32, mode as i8);
            }
        }
    }

    sc
}

// ================================================================
// tr_vinsideT() — C truncyk.c:4083-4224
// Fill (with shadow) via tr_vinside, then trace the V-problem parse into `tr`.
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_vinsidet(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    i1: i32,
    j1: i32,
    j0: i32,
    use_el: bool,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    z_allow_j: bool,
    z_allow_l: bool,
    z_allow_r: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
) -> f32 {
    // r==z: base, just insert the boundary node (C:4099-4107).
    if r == z {
        let mode = if r_allow_j {
            MODE_J
        } else if r_allow_l {
            MODE_L
        } else {
            MODE_R
        };
        insert_trace_node_mode(tr, tr.n - 1, true, i0, j0, r as i32, mode as i8);
        return 0.0;
    }

    let m = cm.m as usize;
    let mut alpha = AlphaMats::new(m);
    let mut shadow = ShadowMats::new(m);
    let ret = tr_vinside(
        cm, dsq, l, r, z, i0, i1, j1, j0, use_el, false, r == 0, z_allow_j, r_allow_l, r_allow_r,
        z_allow_j, z_allow_l, z_allow_r, lm, rm, &mut alpha, Some(&mut shadow),
    );
    let sc = ret.sc;
    let mut mode = ret.mode;
    let mut v = ret.v as usize;
    let mut i = ret.i;
    let mut j = ret.j;

    if r == 0 {
        insert_trace_node_mode(tr, tr.n - 1, true, i, j, v as i32, mode as i8);
    }

    if r != 0 && r != v {
        v = r;
        i = i0;
        j = j0;
        let ip = 0usize;
        let jp = (j0 - j1) as usize;
        mode = MODE_J;
        let aj = alpha.j[v].as_ref().unwrap()[jp][ip];
        let al = alpha.l[v].as_ref().unwrap()[jp][ip];
        let ar = alpha.r[v].as_ref().unwrap()[jp][ip];
        if al > aj {
            mode = MODE_L;
        }
        if ar > aj && ar > al {
            mode = MODE_R;
        }
    }

    // traceback (C:4137-4200)
    while v != z {
        let jp = (j - j1) as usize;
        let ip = (i - i0) as usize;
        let stt = cm.sttype[v] as i32;
        let (yoffset, nxtmode) = if mode == MODE_J {
            (shadow.j[v].as_ref().unwrap()[jp][ip], MODE_J)
        } else if mode == MODE_L {
            (
                shadow.l[v].as_ref().unwrap()[jp][ip],
                shadow.lmode[v].as_ref().unwrap()[jp][ip],
            )
        } else if mode == MODE_R {
            (
                shadow.r[v].as_ref().unwrap()[jp][ip],
                shadow.rmode[v].as_ref().unwrap()[jp][ip],
            )
        } else {
            panic!("tr_vinsidet: unknown mode {mode}");
        };
        match stt {
            S_ST | D_ST => {}
            MP_ST => {
                if mode == MODE_J || mode == MODE_L {
                    i += 1;
                }
                if mode == MODE_J || mode == MODE_R {
                    j -= 1;
                }
            }
            ML_ST | IL_ST => {
                if mode == MODE_J || mode == MODE_L {
                    i += 1;
                }
            }
            MR_ST | IR_ST => {
                if mode == MODE_J || mode == MODE_R {
                    j -= 1;
                }
            }
            _ => panic!("tr_vinsidet: inconceivable state type {stt}"),
        }
        mode = nxtmode;

        if yoffset == USED_EL {
            v = m;
            insert_trace_node_mode(tr, tr.n - 1, true, i, j, v as i32, mode as i8);
            break;
        } else if yoffset == USED_LOCAL_BEGIN {
            panic!("tr_vinsidet: impossible local begin in traceback");
        } else {
            v = (cm.cfirst[v] + yoffset) as usize;
            insert_trace_node_mode(tr, tr.n - 1, true, i, j, v as i32, mode as i8);
        }
    }

    if use_el {
        match cm.sttype[z] as i32 {
            MP_ST => {
                i += 1;
                j -= 1;
            }
            ML_ST | IL_ST => {
                i += 1;
            }
            MR_ST | IR_ST => {
                j -= 1;
            }
            _ => {}
        }
        insert_trace_node_mode(tr, tr.n - 1, true, i, j, m as i32, MODE_J as i8);
    }

    sc
}

// ================================================================
// tr_generic_splitter() — C truncyk.c:500-817
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_generic_splitter(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    j0: i32,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
) -> f32 {
    let m = cm.m as usize;
    let mut v_allow_t = false;
    if r == 0 {
        v_allow_t = true;
    }
    if !r_allow_j && !r_allow_l && !r_allow_r {
        v_allow_t = true;
    }

    // Case 1 (RAMLIMIT==0): direct-solve test always false; skip.
    // Case 2: find first bifurcation r..z-5.
    let mut v = r;
    while (v as i32) <= (z as i32 - 5) {
        if cm.sttype[v] as i32 == B_ST {
            break;
        }
        v += 1;
    }
    // Case 3: no bifurcation -> wedge.
    if cm.sttype[v] as i32 != B_ST {
        return tr_wedge_splitter(cm, dsq, l, tr, r, z, i0, j0, r_allow_j, r_allow_l, r_allow_r, lm, rm);
    }

    let w = cm.cfirst[v] as usize;
    let y = cm.cnum[v] as usize;
    let (wend, yend) = if w < y { (y - 1, z) } else { (z, w - 1) };

    let mut alpha = AlphaMats::new(m);
    let b1 = tr_inside(cm, dsq, l, w, wend, i0, j0, false, r == 0, true, r_allow_l, r_allow_r, false, lm, rm, &mut alpha, None);
    let mut b1_sc = b1.sc;
    let (b1_mode, b1_v, b1_i, b1_j) = (b1.mode, b1.v, b1.i, b1.j);
    if r != 0 {
        b1_sc = IMPOSSIBLE;
    }
    let b2 = tr_inside(cm, dsq, l, y, yend, i0, j0, false, r == 0, true, r_allow_l, r_allow_r, false, lm, rm, &mut alpha, None);
    let mut b2_sc = b2.sc;
    let (b2_mode, b2_v, b2_i, b2_j) = (b2.mode, b2.v, b2.i, b2.j);
    if r != 0 {
        b2_sc = IMPOSSIBLE;
    }

    let mut beta = BetaMats::new(m, l);
    let (b3_sc, b3_mode, b3_v, b3_j) = tr_outside(cm, dsq, l, r, v, i0, j0, false, r_allow_j, r_allow_l, r_allow_r, lm, rm, &mut beta);

    let bigw = j0 - i0 + 1;
    let mut best_sc = IMPOSSIBLE;
    let mut best_k = 0i32;
    let mut best_j = 0i32;
    let mut best_d = 0i32;
    let mut v_mode = 0i32;
    let mut w_mode = 0i32;
    let mut y_mode = 0i32;
    let mut use_el = false;

    let aj_w = alpha.j[w].as_ref().unwrap();
    let al_w = alpha.l[w].as_ref().unwrap();
    let ar_w = alpha.r[w].as_ref().unwrap();
    let aj_y = alpha.j[y].as_ref().unwrap();
    let al_y = alpha.l[y].as_ref().unwrap();
    let ar_y = alpha.r[y].as_ref().unwrap();
    let bj_v = beta.j[v].as_ref().unwrap();
    let bj_el = beta.j[m].as_ref().unwrap();

    for jp in 0..=bigw {
        let j = i0 - 1 + jp;
        let ju = j as usize;
        for d in 0..=jp {
            let du = d as usize;
            for k in 0..=d {
                let ku = k as usize;
                let jk = (j - k) as usize;
                let dk = (d - k) as usize;
                if v_allow_t && k > 0 && k < d {
                    let sc = aj_w[jk][dk] + al_y[ju][ku];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 0; w_mode = 3; y_mode = 2;
                    }
                }
                if v_allow_t && k > 0 && k < d {
                    let sc = ar_w[jk][dk] + aj_y[ju][ku];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 0; w_mode = 1; y_mode = 3;
                    }
                }
                {
                    let sc = aj_w[jk][dk] + aj_y[ju][ku] + bj_v[ju][du];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 3; w_mode = 3; y_mode = 3;
                    }
                }
                if r_allow_l && k > 0 {
                    let sc = aj_w[jk][dk] + al_y[ju][ku] + beta.l[v][(j - d + 1) as usize];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 2; w_mode = 3; y_mode = 2;
                    }
                }
                if r_allow_l && k > 0 {
                    let sc = aj_w[jk][dk] + aj_y[ju][ku] + beta.l[v][(j - d + 1) as usize];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 2; w_mode = 3; y_mode = 3;
                    }
                }
                if r_allow_r && k < d {
                    let sc = ar_w[jk][dk] + aj_y[ju][ku] + beta.r[v][ju];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 1; w_mode = 1; y_mode = 3;
                    }
                }
                if r_allow_r && k < d {
                    let sc = aj_w[jk][dk] + aj_y[ju][ku] + beta.r[v][ju];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 1; w_mode = 3; y_mode = 3;
                    }
                }
                if v_allow_t && k > 0 && k < d {
                    let sc = ar_w[jk][dk] + al_y[ju][ku];
                    if sc > best_sc {
                        best_sc = sc; best_k = k; best_j = j; best_d = d;
                        v_mode = 0; w_mode = 1; y_mode = 2;
                    }
                }
            }

            if r_allow_l {
                let sc = al_w[ju][du] + beta.l[v][(j - d + 1) as usize];
                if sc > best_sc {
                    best_sc = sc; best_k = 0; best_j = j; best_d = d;
                    v_mode = 2; w_mode = 2; y_mode = 0;
                }
            }
            if r_allow_l {
                let sc = aj_w[ju][du] + beta.l[v][(j - d + 1) as usize];
                if sc > best_sc {
                    best_sc = sc; best_k = 0; best_j = j; best_d = d;
                    v_mode = 2; w_mode = 3; y_mode = 0;
                }
            }
            if r_allow_r {
                let sc = ar_y[ju][du] + beta.r[v][ju];
                if sc > best_sc {
                    best_sc = sc; best_k = d; best_j = j; best_d = d;
                    v_mode = 1; w_mode = 0; y_mode = 1;
                }
            }
            if r_allow_r {
                let sc = aj_y[ju][du] + beta.r[v][ju];
                if sc > best_sc {
                    best_sc = sc; best_k = d; best_j = j; best_d = d;
                    v_mode = 1; w_mode = 0; y_mode = 3;
                }
            }
            {
                let sc = bj_el[ju][du];
                if sc > best_sc {
                    best_sc = sc; best_k = -1; best_j = j; best_d = d;
                    v_mode = 3; w_mode = 0; y_mode = 0; use_el = true;
                }
            }
        }
    }

    // r==0 child local entry.
    if r == 0 {
        if b1_sc > best_sc {
            best_sc = b1_sc; best_k = b1_v; best_j = b1_j; best_d = b1_j - b1_i + 1;
            v_mode = 0; w_mode = b1_mode; y_mode = 0;
        }
        if b2_sc > best_sc {
            best_sc = b2_sc; best_k = b2_v; best_j = b2_j; best_d = b2_j - b2_i + 1;
            v_mode = 0; w_mode = 0; y_mode = b2_mode;
        }
    }
    // local hit in parent (marginal).
    if b3_sc > best_sc {
        best_sc = b3_sc; best_k = b3_v; best_j = b3_j;
        v_mode = b3_mode; w_mode = 0; y_mode = 0; use_el = false;
    }

    // Interpret and subdivide.
    if v_mode != 0 {
        if w_mode == MODE_T && y_mode == MODE_T {
            let zsplit = if use_el { v } else { b3_v as usize };
            tr_v_splitter(cm, dsq, l, tr, r, zsplit, i0, best_j, best_j, j0, use_el,
                r_allow_j, r_allow_l, r_allow_r, v_mode == MODE_J, v_mode == MODE_L, v_mode == MODE_R, lm, rm);
            return best_sc;
        } else {
            tr_v_splitter(cm, dsq, l, tr, r, v, i0, best_j - best_d + 1, best_j, j0, false,
                r_allow_j, r_allow_l, r_allow_r, v_mode == MODE_J, v_mode == MODE_L, v_mode == MODE_R, lm, rm);
        }
    } else if w_mode == MODE_T || y_mode == MODE_T {
        if b1_sc > b2_sc {
            insert_trace_node_mode(tr, tr.n - 1, true, b1_i, b1_j, b1_v, b1_mode as i8);
            let z2 = subtree_find_end(cm, b1_v as usize);
            tr_generic_splitter(cm, dsq, l, tr, b1_v as usize, z2, b1_i, b1_j,
                b1_mode == MODE_J, b1_mode == MODE_L, b1_mode == MODE_R, lm, rm);
            return best_sc;
        } else {
            insert_trace_node_mode(tr, tr.n - 1, true, b2_i, b2_j, b2_v, b2_mode as i8);
            let z2 = subtree_find_end(cm, b2_v as usize);
            tr_generic_splitter(cm, dsq, l, tr, b2_v as usize, z2, b2_i, b2_j,
                b2_mode == MODE_J, b2_mode == MODE_L, b2_mode == MODE_R, lm, rm);
            return best_sc;
        }
    } else {
        // case T: parent empty, both children non-empty.
        insert_trace_node_mode(tr, tr.n - 1, true, best_j - best_d + 1, best_j, v as i32, MODE_T as i8);
    }

    let tv = tr.n - 1;
    if w_mode != 0 {
        insert_trace_node_mode(tr, tv, true, best_j - best_d + 1, best_j - best_k, w as i32, w_mode as i8);
        tr_generic_splitter(cm, dsq, l, tr, w, wend, best_j - best_d + 1, best_j - best_k,
            w_mode == MODE_J, w_mode == MODE_L, w_mode == MODE_R, lm, rm);
    } else {
        insert_trace_node_mode(tr, tr.n - 1, true, best_j - best_d + 1, best_j - best_d, w as i32, w_mode as i8);
    }

    if y_mode != 0 {
        insert_trace_node_mode(tr, tv, false, best_j - best_k + 1, best_j, y as i32, y_mode as i8);
        tr_generic_splitter(cm, dsq, l, tr, y, yend, best_j - best_k + 1, best_j,
            y_mode == MODE_J, y_mode == MODE_L, y_mode == MODE_R, lm, rm);
    } else {
        insert_trace_node_mode(tr, tv, false, best_j + 1, best_j, y as i32, y_mode as i8);
    }

    best_sc
}

// ================================================================
// tr_wedge_splitter() — C truncyk.c:829-1013
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_wedge_splitter(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    j0: i32,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
) -> f32 {
    let m = cm.m as usize;
    // base case (RAMLIMIT==0): adjacent nodes only.
    if cm.ndidx[z] == cm.ndidx[r] + 1 {
        return tr_insidet(cm, dsq, l, tr, r, z, i0, j0, r_allow_j, r_allow_l, r_allow_r, false, lm, rm);
    }

    let midnode = cm.ndidx[r] + (cm.ndidx[z] - cm.ndidx[r]) / 2;
    let w = cm.nodemap[midnode as usize] as usize;
    let y = (cm.cfirst[w] - 1) as usize;

    let mut alpha = AlphaMats::new(m);
    let b1 = tr_inside(cm, dsq, l, w, z, i0, j0, false, r == 0, true, r_allow_l, r_allow_r, false, lm, rm, &mut alpha, None);
    let mut b1_sc = b1.sc;
    let (b1_mode, b1_v, b1_i, b1_j) = (b1.mode, b1.v, b1.i, b1.j);
    if r != 0 {
        b1_sc = IMPOSSIBLE;
    }
    let mut beta = BetaMats::new(m, l);
    let (mut b2_sc, b2_mode, b2_v, b2_j) = tr_outside(cm, dsq, l, r, y, i0, j0, false, r_allow_j, r_allow_l, r_allow_r, lm, rm, &mut beta);
    if b2_mode == MODE_L && !r_allow_l {
        b2_sc = IMPOSSIBLE;
    }
    if b2_mode == MODE_R && !r_allow_r {
        b2_sc = IMPOSSIBLE;
    }

    let bigw = j0 - i0 + 1;
    let mut best_sc = IMPOSSIBLE;
    let mut best_v = 0i32;
    let mut best_j = 0i32;
    let mut best_d = 0i32;
    let mut p_mode = 0i32;
    let mut c_mode = 0i32;

    if b1_sc > best_sc {
        best_sc = b1_sc; best_v = b1_v; best_j = b1_j; best_d = b1_j - b1_i + 1;
        p_mode = 0; c_mode = b1_mode;
    }
    if b2_sc > best_sc {
        best_sc = b2_sc; best_v = b2_v; best_j = b2_j; best_d = 1;
        p_mode = b2_mode; c_mode = 0;
    }

    for vv in w..=y {
        let ajv = alpha.j[vv].as_ref().unwrap();
        let alv = alpha.l[vv].as_ref().unwrap();
        let arv = alpha.r[vv].as_ref().unwrap();
        let bjv = beta.j[vv].as_ref().unwrap();
        for jp in 0..=bigw {
            let j = i0 - 1 + jp;
            let ju = j as usize;
            for d in 0..=jp {
                let du = d as usize;
                let sc = ajv[ju][du] + bjv[ju][du];
                if sc > best_sc {
                    best_sc = sc; best_v = vv as i32; best_d = d; best_j = j; p_mode = 3; c_mode = 3;
                }
                if r_allow_l {
                    let sc = ajv[ju][du] + beta.l[vv][(j - d + 1) as usize];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_d = d; best_j = j; p_mode = 2; c_mode = 3;
                    }
                }
                if r_allow_l {
                    let sc = alv[ju][du] + beta.l[vv][(j - d + 1) as usize];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_d = d; best_j = j; p_mode = 2; c_mode = 2;
                    }
                }
                if r_allow_r {
                    let sc = ajv[ju][du] + beta.r[vv][ju];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_d = d; best_j = j; p_mode = 1; c_mode = 3;
                    }
                }
                if r_allow_r {
                    let sc = arv[ju][du] + beta.r[vv][ju];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_d = d; best_j = j; p_mode = 1; c_mode = 1;
                    }
                }
            }
        }
    }

    // Joint parent to EL.
    let bj_el = beta.j[m].as_ref().unwrap();
    for jp in 0..=bigw {
        let j = i0 - 1 + jp;
        let ju = j as usize;
        for d in 0..=jp {
            let sc = bj_el[ju][d as usize];
            if sc > best_sc {
                best_sc = sc; best_v = -1; best_j = j; best_d = d; p_mode = 3; c_mode = 0;
            }
        }
    }

    if p_mode != 0 {
        if c_mode == MODE_T {
            let zsplit = if p_mode == MODE_J { w } else { b2_v as usize };
            tr_v_splitter(cm, dsq, l, tr, r, zsplit, i0, best_j - best_d + 1, best_j, j0,
                p_mode == MODE_J, r_allow_j, r_allow_l, r_allow_r, p_mode == MODE_J, p_mode == MODE_L, p_mode == MODE_R, lm, rm);
            return best_sc;
        } else {
            tr_v_splitter(cm, dsq, l, tr, r, best_v as usize, i0, best_j - best_d + 1, best_j, j0, false,
                r_allow_j, r_allow_l, r_allow_r, c_mode == MODE_J, c_mode == MODE_L, c_mode == MODE_R, lm, rm);
        }
    }

    if c_mode != 0 {
        if p_mode == MODE_T {
            insert_trace_node_mode(tr, tr.n - 1, true, best_j - best_d + 1, best_j, best_v, c_mode as i8);
        }
        tr_wedge_splitter(cm, dsq, l, tr, best_v as usize, z, best_j - best_d + 1, best_j,
            c_mode == MODE_J, c_mode == MODE_L, c_mode == MODE_R, lm, rm);
    } else {
        panic!("tr_wedge_splitter: p_mode={p_mode} c_mode={c_mode}");
    }

    best_sc
}

// ================================================================
// tr_v_splitter() — C truncyk.c:1024-1205
// ================================================================
#[allow(clippy::too_many_arguments)]
pub(crate) fn tr_v_splitter(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    i1: i32,
    j1: i32,
    j0: i32,
    use_el: bool,
    r_allow_j: bool,
    r_allow_l: bool,
    r_allow_r: bool,
    z_allow_j: bool,
    z_allow_l: bool,
    z_allow_r: bool,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
) {
    let m = cm.m as usize;
    // base case (RAMLIMIT==0): adjacent nodes or r==z.
    if cm.ndidx[z] == cm.ndidx[r] + 1 || r == z {
        tr_vinsidet(cm, dsq, l, tr, r, z, i0, i1, j1, j0, use_el,
            r_allow_j, r_allow_l, r_allow_r, z_allow_j, z_allow_l, z_allow_r, lm, rm);
        return;
    }

    let midnode = cm.ndidx[r] + (cm.ndidx[z] - cm.ndidx[r]) / 2;
    let w = cm.nodemap[midnode as usize] as usize;
    let y = (cm.cfirst[w] - 1) as usize;

    let mut alpha = AlphaMats::new(m);
    // NB C passes tr_vinside's r_allow_J = z_allow_J.
    let b = tr_vinside(cm, dsq, l, w, z, i0, i1, j1, j0, use_el, false, r == 0,
        z_allow_j, r_allow_l, r_allow_r, z_allow_j, z_allow_l, z_allow_r, lm, rm, &mut alpha, None);
    let mut b_sc = b.sc;
    let (b_mode, b_v, b_i, b_j) = (b.mode, b.v, b.i, b.j);
    if r != 0 {
        b_sc = IMPOSSIBLE;
    }
    let mut beta = BetaMats::new(m, l);
    tr_voutside(cm, dsq, l, r, y, i0, i1, j1, j0, use_el, false,
        r_allow_j, r_allow_l, r_allow_r, z_allow_j, z_allow_l, z_allow_r, lm, rm, &mut beta);

    let mut best_sc = IMPOSSIBLE;
    let mut best_v = 0i32;
    let mut best_i = 0i32;
    let mut best_j = 0i32;
    let mut p_mode = 0i32;
    let mut c_mode = 0i32;

    if b_sc > best_sc {
        best_sc = b_sc; best_v = b_v; best_i = b_i; best_j = b_j; p_mode = 0; c_mode = b_mode;
    }

    let ispan = (i1 - i0) as usize;
    let jspan = (j0 - j1) as usize;
    for vv in w..=y {
        let ajv = alpha.j[vv].as_ref().unwrap();
        let alv = alpha.l[vv].as_ref().unwrap();
        let arv = alpha.r[vv].as_ref().unwrap();
        let bjv = beta.j[vv].as_ref().unwrap();
        for ip in 0..=ispan {
            for jp in 0..=jspan {
                if z_allow_j {
                    let sc = ajv[jp][ip] + bjv[jp][ip];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_i = ip as i32 + i0; best_j = jp as i32 + j1; p_mode = 3; c_mode = 3;
                    }
                }
                if z_allow_j && r_allow_l {
                    let sc = ajv[jp][ip] + beta.l[vv][ip];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_i = ip as i32 + i0; best_j = jp as i32 + j1; p_mode = 2; c_mode = 3;
                    }
                }
                if r_allow_l {
                    let sc = alv[jp][ip] + beta.l[vv][ip];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_i = ip as i32 + i0; best_j = jp as i32 + j1; p_mode = 2; c_mode = 2;
                    }
                }
                if z_allow_j && r_allow_r {
                    let sc = ajv[jp][ip] + beta.r[vv][jp];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_i = ip as i32 + i0; best_j = jp as i32 + j1; p_mode = 1; c_mode = 3;
                    }
                }
                if r_allow_r {
                    let sc = arv[jp][ip] + beta.r[vv][jp];
                    if sc > best_sc {
                        best_sc = sc; best_v = vv as i32; best_i = ip as i32 + i0; best_j = jp as i32 + j1; p_mode = 1; c_mode = 1;
                    }
                }
            }
        }
    }

    if use_el {
        let bj_el = beta.j[m].as_ref().unwrap();
        for ip in 0..=ispan {
            for jp in 0..=jspan {
                let sc = bj_el[jp][ip];
                if sc > best_sc {
                    best_sc = sc; best_v = m as i32; best_i = ip as i32 + i0; best_j = jp as i32 + j1; p_mode = 3; c_mode = 0;
                }
            }
        }
    }

    if p_mode != 0 {
        if c_mode != 0 {
            tr_v_splitter(cm, dsq, l, tr, r, best_v as usize, i0, best_i, best_j, j0, false,
                r_allow_j, r_allow_l, r_allow_r, c_mode == MODE_J, c_mode == MODE_L, c_mode == MODE_R, lm, rm);
            tr_v_splitter(cm, dsq, l, tr, best_v as usize, z, best_i, i1, j1, best_j, use_el,
                c_mode == MODE_J, c_mode == MODE_L, c_mode == MODE_R, z_allow_j, z_allow_l, z_allow_r, lm, rm);
        } else {
            tr_v_splitter(cm, dsq, l, tr, r, w, i0, best_i, best_j, j0, true,
                r_allow_j, r_allow_l, r_allow_r, true, false, false, lm, rm);
        }
    } else {
        if best_v != z as i32 {
            insert_trace_node_mode(tr, tr.n - 1, true, best_i, best_j, best_v, c_mode as i8);
        }
        tr_v_splitter(cm, dsq, l, tr, best_v as usize, z, best_i, i1, j1, best_j, use_el,
            c_mode == MODE_J, c_mode == MODE_L, c_mode == MODE_R, z_allow_j, z_allow_l, z_allow_r, lm, rm);
    }
}

// ================================================================
// TrCYK_DnC() — C truncyk.c:322-385 (do_1p0 == FALSE path only)
// Top-level truncated divide-and-conquer CYK alignment. Called with r=0, i0=1,
// j0=L (cm_alndata.c:379). Returns (parsetree, score_bits, overall_mode).
// ================================================================
#[allow(clippy::too_many_arguments)]
pub fn tr_cyk_dnc(
    cm: &CM,
    trp: &crate::cm_trunc::TrPenalties,
    lm: &[Vec<f32>],
    rm: &[Vec<f32>],
    dsq: &[u8],
    l: i32,
    pass_idx: i32,
    use_local: bool,
) -> (Parsetree, f32, i32) {
    let i0 = 1;
    let j0 = l;
    let mut tr = Parsetree::new(100);
    tr.is_std = false;
    tr.pass_idx = pass_idx;
    // C:344 InsertTraceNode(tr,-1,LEFT,i0,j0,0) — state 0, default mode J.
    insert_trace_node_mode(&mut tr, -1, true, i0, j0, 0, MODE_J as i8);
    let z = cm.m as usize - 1;

    let sc = tr_generic_splitter(cm, dsq, l, &mut tr, 0, z, i0, j0, true, true, true, lm, rm);

    // r==0: adopt the full-alignment mode into the root node (C:359-361).
    tr.mode[0] = tr.mode[1];

    // Truncated-begin penalty (C truncyk.c:375-379).
    let pty = trp.pty_slice(pass_idx, use_local);
    let bsc = pty[tr.state[1] as usize];
    tr.trpenalty = bsc; // C truncyk.c:378: tr->trpenalty = bsc
    let total = sc + bsc;
    let mode = tr.mode[0] as i32;
    (tr, total, mode)
}



