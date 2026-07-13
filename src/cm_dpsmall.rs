//! Faithful port of C `cm_dpsmall.c` `CYKDivideAndConquer` — the memory-efficient
//! divide-and-conquer *exact* CYK alignment used by
//! `cmalign --small --cyk --noprob --nonbanded --notrunc`.
//!
//! This ports the non-QDB / non-banded / non-truncated path only (dmin==dmax==NULL,
//! `generic_splitter`), which is what `cmalign --small` dispatches to
//! (`cm_alndata.c:386` `CYKDivideAndConquer(cm,dsq,L,0,1,L,&tr,NULL,NULL)`).
//!
//! Structure mirrors C exactly:
//!   * [`cyk_divide_and_conquer`]  — C `CYKDivideAndConquer` (cm_dpsmall.c:302)
//!   * [`generic_splitter`]        — C `generic_splitter`     (cm_dpsmall.c:686)
//!   * [`wedge_splitter`]          — C `wedge_splitter`       (cm_dpsmall.c:907)
//!   * [`v_splitter`]              — C `v_splitter`           (cm_dpsmall.c:1086)
//!   * [`inside`]/[`outside`]      — vjd CYK engines          (cm_dpsmall.c:1321/1689)
//!   * [`vinside`]/[`voutside`]    — vji CYK engines          (cm_dpsmall.c:2088/2480)
//!   * [`inside_t`]/[`vinside_t`]  — fill+traceback           (cm_dpsmall.c:2806/2943)
//!
//! Note on memory: C's `deckpool_s`/`touch`/`nends` machinery is a pure memory
//! optimization (deck reuse); it has *no* effect on the computed values or the
//! output parsetree. We use straightforward per-deck `Vec` allocations retained
//! for the lifetime of each engine call, which is byte-output-identical. The D&C
//! split structure itself (which bounds peak memory) is ported faithfully.
//!
//! C's `insideT_size(...) < RAMLIMIT` "solve directly if small" test is a no-op
//! because `RAMLIMIT == 0` (config.h): a positive size is never `< 0`, so the
//! recursion only bottoms out at the structural boundaries `ndidx[z]==ndidx[r]+1`
//! (adjacent nodes) or `r==z`. We reproduce exactly those boundary tests.

use crate::cm::CM;
use crate::constants::{B_ST, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, ROOT_S, S_ST};
use crate::parsetree::Parsetree;

const IMPOSSIBLE: f32 = -1.0e36;
const USED_LOCAL_BEGIN: i32 = 101;
const USED_EL: i32 = 102;
const CMH_LOCAL_BEGIN: u32 = 1 << 10;
const CMH_LOCAL_END: u32 = 1 << 11;
const K: usize = 4;

#[inline]
fn not_impossible(x: f32) -> bool {
    x > -0.5e36
}

/// C `StateDelta` (cm.c): number of residues emitted by a state.
#[inline]
fn state_delta(stt: i32) -> i32 {
    match stt {
        MP_ST => 2,
        ML_ST | MR_ST | IL_ST | IR_ST => 1,
        _ => 0,
    }
}

/// C singlet emission with degenerate averaging: `dsq<K ? esc[dsq] : FAvgScore`.
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

/// C pair emission with degenerate averaging: `DegeneratePairScore`.
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

/// C `CMSubtreeFindEnd` (cm.c:1013): find the END_E state that closes the subtree
/// rooted at `r`.
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

/// C `InsertTraceNode` (parsetree.c): append a node whose parent is `parent`
/// (attach as left or right child), returning its index.
fn insert_trace_node(
    tr: &mut Parsetree,
    parent: i32,
    is_left: bool,
    emitl: i32,
    emitr: i32,
    state: i32,
) -> i32 {
    let idx = tr.add_node(emitl, emitr, state, -1, -1, parent);
    if parent >= 0 {
        if is_left {
            tr.nxtl[parent as usize] = idx;
        } else {
            tr.nxtr[parent as usize] = idx;
        }
    }
    idx
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

// ---- vji deck management (coords [v][jp][ip], jp 0..j0-j1, ip 0..i1-i0) ----
type VjiDeck = Vec<Vec<f32>>;
type VjiShadow = Vec<Vec<i32>>;

fn new_vji_deck(i0: i32, i1: i32, j1: i32, j0: i32, init: f32) -> VjiDeck {
    vec![vec![init; (i1 - i0 + 1) as usize]; (j0 - j1 + 1) as usize]
}
fn new_vji_shadow(i0: i32, i1: i32, j1: i32, j0: i32, init: i32) -> VjiShadow {
    vec![vec![init; (i1 - i0 + 1) as usize]; (j0 - j1 + 1) as usize]
}

// ================================================================
// inside() — C cm_dpsmall.c:1321
// Fills alpha[vroot..vend] (vjd) into `alpha`. Returns (shadow?, b, bsc).
// ================================================================
#[allow(clippy::too_many_arguments)]
fn inside(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    vroot: usize,
    vend: usize,
    i0: i32,
    j0: i32,
    alpha: &mut Vec<Option<VjdDeck>>,
    want_shadow: bool,
    allow_begin: bool,
) -> (Option<Vec<Option<VjdShadow>>>, i32, f32) {
    let mut b: i32 = -1;
    let mut bsc = IMPOSSIBLE;
    let w = j0 - i0 + 1;
    let m = cm.m as usize;
    let mut shadow: Vec<Option<VjdShadow>> = if want_shadow {
        (0..=m).map(|_| None).collect()
    } else {
        Vec::new()
    };

    for v in (vroot..=vend).rev() {
        let stt = cm.sttype[v] as i32;
        if stt == E_ST {
            // C: reuse the (shared) 'end' deck: end[j][0]=0, end[j][d>=1]=IMPOSSIBLE
            let mut deck = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
            for jp in 0..=w {
                let j = (i0 + jp - 1) as usize;
                deck[j][0] = 0.0;
            }
            alpha[v] = Some(deck);
            continue;
        }
        let mut deck = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        let mut sh: VjdShadow = if want_shadow {
            new_vjd_shadow(l, i0, j0, USED_EL)
        } else {
            Vec::new()
        };
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let endsc_v = cm.endsc[v];
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];
        let sd = state_delta(stt);

        if stt == D_ST || stt == S_ST {
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                for d in 0..=(jp as usize) {
                    let mut val = endsc_v + cm.el_selfsc * ((d as i32 - sd) as f32);
                    let mut ysh = USED_EL;
                    for yoffset in 0..cnum {
                        let sc = alpha[cfirst + yoffset].as_ref().unwrap()[j][d] + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    deck[j][d] = val;
                    if want_shadow {
                        sh[j][d] = ysh;
                    }
                }
            }
        } else if stt == B_ST {
            let y = cfirst;
            let z = cnum; // for B, cnum[v] holds right child state index
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                for d in 0..=(jp as usize) {
                    let mut val = alpha[y].as_ref().unwrap()[j][d] + alpha[z].as_ref().unwrap()[j][0];
                    let mut ksh = 0i32;
                    for k in 1..=d {
                        let sc = alpha[y].as_ref().unwrap()[j - k][d - k]
                            + alpha[z].as_ref().unwrap()[j][k];
                        if sc > val {
                            val = sc;
                            ksh = k as i32;
                        }
                    }
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    deck[j][d] = val;
                    if want_shadow {
                        sh[j][d] = ksh;
                    }
                }
            }
        } else if stt == MP_ST {
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                deck[j][0] = IMPOSSIBLE;
                if jp > 0 {
                    deck[j][1] = IMPOSSIBLE;
                }
                for d in 2..=(jp as usize) {
                    let mut val = endsc_v + cm.el_selfsc * ((d as i32 - sd) as f32);
                    let mut ysh = USED_EL;
                    for yoffset in 0..cnum {
                        let sc =
                            alpha[cfirst + yoffset].as_ref().unwrap()[j - 1][d - 2] + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    let i = j - d + 1;
                    val += pair_sc(esc_v, dsq[i], dsq[j]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    deck[j][d] = val;
                    if want_shadow {
                        sh[j][d] = ysh;
                    }
                }
            }
        } else if stt == IL_ST || stt == ML_ST {
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                deck[j][0] = IMPOSSIBLE;
                for d in 1..=(jp as usize) {
                    let mut val = endsc_v + cm.el_selfsc * ((d as i32 - sd) as f32);
                    let mut ysh = USED_EL;
                    for yoffset in 0..cnum {
                        // IL self-transits (cfirst[v]==v); read the in-progress deck.
                        let child = cfirst + yoffset;
                        let cv = if child == v {
                            deck[j][d - 1]
                        } else {
                            alpha[child].as_ref().unwrap()[j][d - 1]
                        };
                        let sc = cv + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    let i = j - d + 1;
                    val += sing_sc(esc_v, dsq[i]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    deck[j][d] = val;
                    if want_shadow {
                        sh[j][d] = ysh;
                    }
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                deck[j][0] = IMPOSSIBLE;
                for d in 1..=(jp as usize) {
                    let mut val = endsc_v + cm.el_selfsc * ((d as i32 - sd) as f32);
                    let mut ysh = USED_EL;
                    for yoffset in 0..cnum {
                        // IR self-transits (cfirst[v]==v); read the in-progress deck.
                        let child = cfirst + yoffset;
                        let cv = if child == v {
                            deck[j - 1][d - 1]
                        } else {
                            alpha[child].as_ref().unwrap()[j - 1][d - 1]
                        };
                        let sc = cv + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    val += sing_sc(esc_v, dsq[j]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    deck[j][d] = val;
                    if want_shadow {
                        sh[j][d] = ysh;
                    }
                }
            }
        }

        alpha[v] = Some(deck);
        if want_shadow {
            shadow[v] = Some(sh);
        }

        // local begin bookkeeping (C cm_dpsmall.c:1596)
        if allow_begin {
            let av = alpha[v].as_ref().unwrap()[j0 as usize][w as usize] + cm.beginsc[v];
            if av > bsc {
                b = v as i32;
                bsc = av;
            }
        }
        if allow_begin && v == 0 && bsc > alpha[0].as_ref().unwrap()[j0 as usize][w as usize] {
            alpha[0].as_mut().unwrap()[j0 as usize][w as usize] = bsc;
            if want_shadow {
                shadow[0].as_mut().unwrap()[j0 as usize][w as usize] = USED_LOCAL_BEGIN;
            }
        }
    }

    if want_shadow {
        (Some(shadow), b, bsc)
    } else {
        (None, b, bsc)
    }
}

// ================================================================
// outside() — C cm_dpsmall.c:1689
// Fills beta[vroot's split set..vend] (+ EL deck at M) into `beta`.
// ================================================================
#[allow(clippy::too_many_arguments)]
fn outside(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    vroot: usize,
    vend: usize,
    i0: i32,
    j0: i32,
    beta: &mut Vec<Option<VjdDeck>>,
) {
    let m = cm.m as usize;
    let w = j0 - i0 + 1;
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;

    // Initialize the root deck / split set (C:1723).
    let w1 = cm.nodemap[cm.ndidx[vroot] as usize] as usize;
    let w2 = if cm.sttype[vroot] as i32 == B_ST {
        w1
    } else {
        (cm.cfirst[w1] - 1) as usize
    };
    for v in w1..=w2 {
        beta[v] = Some(new_vjd_deck(l, i0, j0, IMPOSSIBLE));
    }
    beta[vroot].as_mut().unwrap()[j0 as usize][w as usize] = 0.0;

    // EL deck init + vroot->EL unroll (C:1741).
    if local_end {
        let mut eld = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        if not_impossible(cm.endsc[vroot]) {
            let esc_v = &cm.esc[vroot];
            match cm.sttype[vroot] as i32 {
                MP_ST => {
                    if w >= 2 {
                        let escore = pair_sc(esc_v, dsq[i0 as usize], dsq[j0 as usize]);
                        let mut val =
                            cm.endsc[vroot] + cm.el_selfsc * ((w - 2) as f32) + escore;
                        if val < IMPOSSIBLE {
                            val = IMPOSSIBLE;
                        }
                        eld[(j0 - 1) as usize][(w - 2) as usize] = val;
                    }
                }
                ML_ST | IL_ST => {
                    if w >= 1 {
                        let escore = sing_sc(esc_v, dsq[i0 as usize]);
                        let mut val =
                            cm.endsc[vroot] + cm.el_selfsc * ((w - 1) as f32) + escore;
                        if val < IMPOSSIBLE {
                            val = IMPOSSIBLE;
                        }
                        eld[j0 as usize][(w - 1) as usize] = val;
                    }
                }
                MR_ST | IR_ST => {
                    if w >= 1 {
                        let escore = sing_sc(esc_v, dsq[j0 as usize]);
                        let mut val =
                            cm.endsc[vroot] + cm.el_selfsc * ((w - 1) as f32) + escore;
                        if val < IMPOSSIBLE {
                            val = IMPOSSIBLE;
                        }
                        eld[(j0 - 1) as usize][(w - 1) as usize] = val;
                    }
                }
                S_ST | D_ST => {
                    let mut val = cm.endsc[vroot] + cm.el_selfsc * (w as f32);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    eld[j0 as usize][w as usize] = val;
                }
                _ => {}
            }
        }
        beta[m] = Some(eld);
    }

    // Main loop down through the decks (C:1817).
    for v in (w2 + 1)..=vend {
        let mut deck = new_vjd_deck(l, i0, j0, IMPOSSIBLE);
        // local begin into v (C:1834)
        if vroot == 0 && i0 == 1 && j0 == l && local_begin {
            deck[j0 as usize][w as usize] = cm.beginsc[v];
        }
        let plast = cm.plast[v];
        let pnum = cm.pnum[v];
        for jp in (0..=w).rev() {
            let j = (i0 - 1 + jp) as usize;
            for d in (0..=jp).rev() {
                let d = d as usize;
                let i = j - d + 1;
                let mut best = deck[j][d];
                let mut y = plast;
                while y > plast - pnum {
                    let yu = y as usize;
                    if (yu as i32) < vroot as i32 {
                        y -= 1;
                        continue;
                    }
                    let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                    let yst = cm.sttype[yu] as i32;
                    let esc_y = &cm.esc[yu];
                    let tsc_y = &cm.tsc[yu];
                    // IL/IR self-loop: parent y may be v itself; its deck is the
                    // in-progress `deck` (not yet in the matrix).
                    let is_self = yu == v;
                    match yst {
                        x if x == MP_ST => {
                            if !(j as i32 == j0 || d as i32 == jp) {
                                let escore = pair_sc(esc_y, dsq[i - 1], dsq[j + 1]);
                                let pv = if is_self {
                                    deck[j + 1][d + 2]
                                } else {
                                    beta[yu].as_ref().unwrap()[j + 1][d + 2]
                                };
                                let sc = pv + tsc_y[voffset] + escore;
                                if sc > best {
                                    best = sc;
                                }
                            }
                        }
                        x if x == ML_ST || x == IL_ST => {
                            if d as i32 != jp {
                                let escore = sing_sc(esc_y, dsq[i - 1]);
                                let pv = if is_self {
                                    deck[j][d + 1]
                                } else {
                                    beta[yu].as_ref().unwrap()[j][d + 1]
                                };
                                let sc = pv + tsc_y[voffset] + escore;
                                if sc > best {
                                    best = sc;
                                }
                            }
                        }
                        x if x == MR_ST || x == IR_ST => {
                            if j as i32 != j0 {
                                let escore = sing_sc(esc_y, dsq[j + 1]);
                                let pv = if is_self {
                                    deck[j + 1][d + 1]
                                } else {
                                    beta[yu].as_ref().unwrap()[j + 1][d + 1]
                                };
                                let sc = pv + tsc_y[voffset] + escore;
                                if sc > best {
                                    best = sc;
                                }
                            }
                        }
                        _ => {
                            // S, E, D
                            let pv = if is_self {
                                deck[j][d]
                            } else {
                                beta[yu].as_ref().unwrap()[j][d]
                            };
                            let sc = pv + tsc_y[voffset];
                            if sc > best {
                                best = sc;
                            }
                        }
                    }
                    y -= 1;
                }
                if best < IMPOSSIBLE {
                    best = IMPOSSIBLE;
                }
                deck[j][d] = best;
            }
        }
        beta[v] = Some(deck);

        // v -> EL local end transitions (C:1898)
        if local_end && not_impossible(cm.endsc[v]) {
            let vst = cm.sttype[v] as i32;
            let esc_v = &cm.esc[v];
            let endsc_v = cm.endsc[v];
            for jp in 0..=w {
                let j = (i0 - 1 + jp) as usize;
                for d in 0..=(jp as usize) {
                    let i = j - d + 1;
                    let cand = match vst {
                        x if x == MP_ST => {
                            if j as i32 == j0 || d as i32 == jp {
                                None
                            } else {
                                let escore = pair_sc(esc_v, dsq[i - 1], dsq[j + 1]);
                                Some(
                                    beta[v].as_ref().unwrap()[j + 1][d + 2]
                                        + endsc_v
                                        + cm.el_selfsc * (d as f32)
                                        + escore,
                                )
                            }
                        }
                        x if x == ML_ST || x == IL_ST => {
                            if d as i32 == jp {
                                None
                            } else {
                                let escore = sing_sc(esc_v, dsq[i - 1]);
                                Some(
                                    beta[v].as_ref().unwrap()[j][d + 1]
                                        + endsc_v
                                        + cm.el_selfsc * (d as f32)
                                        + escore,
                                )
                            }
                        }
                        x if x == MR_ST || x == IR_ST => {
                            if j as i32 == j0 {
                                None
                            } else {
                                let escore = sing_sc(esc_v, dsq[j + 1]);
                                Some(
                                    beta[v].as_ref().unwrap()[j + 1][d + 1]
                                        + endsc_v
                                        + cm.el_selfsc * (d as f32)
                                        + escore,
                                )
                            }
                        }
                        _ => {
                            // S, D, E
                            Some(
                                beta[v].as_ref().unwrap()[j][d]
                                    + endsc_v
                                    + cm.el_selfsc * (d as f32),
                            )
                        }
                    };
                    if let Some(sc) = cand {
                        if sc > beta[m].as_ref().unwrap()[j][d] {
                            beta[m].as_mut().unwrap()[j][d] = sc;
                        }
                    }
                }
            }
        }
    }
}

// ================================================================
// vinside() — C cm_dpsmall.c:2088 (vji coords)
// Fills a[r..w2 of z's split set] into `a`. Returns (shadow?, b, bsc).
// ================================================================
#[allow(clippy::too_many_arguments)]
fn vinside(
    cm: &CM,
    dsq: &[u8],
    r: usize,
    z: usize,
    i0: i32,
    i1: i32,
    j1: i32,
    j0: i32,
    use_el: bool,
    a: &mut Vec<Option<VjiDeck>>,
    want_shadow: bool,
    allow_begin: bool,
) -> (Option<Vec<Option<VjiShadow>>>, i32, f32) {
    let m = cm.m as usize;
    let mut b: i32 = -1;
    let mut bsc = IMPOSSIBLE;
    let mut shadow: Vec<Option<VjiShadow>> = if want_shadow {
        (0..=m).map(|_| None).collect()
    } else {
        Vec::new()
    };

    // the whole split set w1<=z<=w2 must be initialized (C:2128)
    let w1 = cm.nodemap[cm.ndidx[z] as usize] as usize;
    let w2 = (cm.cfirst[w1] - 1) as usize;
    for v in w1..=w2 {
        a[v] = Some(new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE));
        if want_shadow {
            shadow[v] = Some(new_vji_shadow(i0, i1, j1, j0, USED_EL));
        }
    }

    // boundary cell init (C:2151)
    let ip0 = (i1 - i0) as usize;
    if !use_el {
        a[z].as_mut().unwrap()[0][ip0] = 0.0;
    } else {
        let esc_z = &cm.esc[z];
        let endsc_z = cm.endsc[z];
        match cm.sttype[z] as i32 {
            x if x == D_ST || x == S_ST => {
                a[z].as_mut().unwrap()[0][ip0] =
                    endsc_z + cm.el_selfsc * ((j1 - (ip0 as i32 + i0) + 1) as f32);
                if want_shadow {
                    shadow[z].as_mut().unwrap()[0][ip0] = USED_EL;
                }
            }
            x if x == MP_ST => {
                if !(i0 == i1 || j1 == j0) {
                    let mut val = endsc_z + cm.el_selfsc * ((j1 - (ip0 as i32 + i0) + 1) as f32);
                    val += pair_sc(esc_z, dsq[(i1 - 1) as usize], dsq[(j1 + 1) as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    a[z].as_mut().unwrap()[1][ip0 - 1] = val;
                    if want_shadow {
                        shadow[z].as_mut().unwrap()[1][ip0 - 1] = USED_EL;
                    }
                }
            }
            x if x == ML_ST || x == IL_ST => {
                if i0 != i1 {
                    let mut val = endsc_z + cm.el_selfsc * ((j1 - (ip0 as i32 + i0) + 1) as f32);
                    val += sing_sc(esc_z, dsq[(i1 - 1) as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    a[z].as_mut().unwrap()[0][ip0 - 1] = val;
                    if want_shadow {
                        shadow[z].as_mut().unwrap()[0][ip0 - 1] = USED_EL;
                    }
                }
            }
            x if x == MR_ST || x == IR_ST => {
                if j1 != j0 {
                    let mut val = endsc_z + cm.el_selfsc * ((j1 - (ip0 as i32 + i0) + 1) as f32);
                    val += sing_sc(esc_z, dsq[(j1 + 1) as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    a[z].as_mut().unwrap()[1][ip0] = val;
                    if want_shadow {
                        shadow[z].as_mut().unwrap()[1][ip0] = USED_EL;
                    }
                }
            }
            _ => {}
        }
    }

    // special empty-seq begin (C:2223)
    if allow_begin && j0 - j1 == 0 && i1 - i0 == 0 {
        b = z as i32;
        bsc = a[z].as_ref().unwrap()[0][0] + cm.beginsc[z];
        if z == 0 {
            a[0].as_mut().unwrap()[0][0] = bsc;
            if want_shadow {
                shadow[0].as_mut().unwrap()[0][0] = USED_LOCAL_BEGIN;
            }
        }
    }

    // Main recursion (C:2236)
    let jpmax = (j0 - j1) as usize;
    let ipmax = (i1 - i0) as usize;
    for v in (r..=(w1 - 1)).rev() {
        a[v] = Some(new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE));
        if want_shadow {
            shadow[v] = Some(new_vji_shadow(i0, i1, j1, j0, USED_EL));
        }
        let stt = cm.sttype[v] as i32;
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let endsc_v = cm.endsc[v];
        let esc_v = &cm.esc[v];
        let tsc_v = &cm.tsc[v];

        if stt == D_ST || stt == S_ST {
            for jp in 0..=jpmax {
                for ip in (0..=ipmax).rev() {
                    let y = cfirst;
                    let mut val = a[y].as_ref().unwrap()[jp][ip] + tsc_v[0];
                    let mut ysh = 0i32;
                    if use_el
                        && not_impossible(endsc_v)
                        && (endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1
                                    - state_delta(stt)) as f32))
                            > val
                    {
                        val = endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1 - state_delta(stt))
                                    as f32);
                        ysh = USED_EL;
                    }
                    for yoffset in 1..cnum {
                        let sc = a[y + yoffset].as_ref().unwrap()[jp][ip] + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    a[v].as_mut().unwrap()[jp][ip] = val;
                    if want_shadow {
                        shadow[v].as_mut().unwrap()[jp][ip] = ysh;
                    }
                }
            }
        } else if stt == MP_ST {
            for ip in (0..=ipmax).rev() {
                a[v].as_mut().unwrap()[0][ip] = IMPOSSIBLE;
            }
            for jp in 1..=jpmax {
                let j = jp as i32 + j1;
                a[v].as_mut().unwrap()[jp][ipmax] = IMPOSSIBLE;
                for ip in (0..ipmax).rev() {
                    let i = ip as i32 + i0;
                    let y = cfirst;
                    let mut val = a[y].as_ref().unwrap()[jp - 1][ip + 1] + tsc_v[0];
                    let mut ysh = 0i32;
                    if use_el
                        && not_impossible(endsc_v)
                        && (endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1
                                    - state_delta(stt)) as f32))
                            > val
                    {
                        val = endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1 - state_delta(stt))
                                    as f32);
                        ysh = USED_EL;
                    }
                    for yoffset in 1..cnum {
                        let sc = a[y + yoffset].as_ref().unwrap()[jp - 1][ip + 1] + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    val += pair_sc(esc_v, dsq[i as usize], dsq[j as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    a[v].as_mut().unwrap()[jp][ip] = val;
                    if want_shadow {
                        shadow[v].as_mut().unwrap()[jp][ip] = ysh;
                    }
                }
            }
        } else if stt == ML_ST || stt == IL_ST {
            for jp in 0..=jpmax {
                a[v].as_mut().unwrap()[jp][ipmax] = IMPOSSIBLE;
                for ip in (0..ipmax).rev() {
                    let i = ip as i32 + i0;
                    let y = cfirst;
                    let mut val = a[y].as_ref().unwrap()[jp][ip + 1] + tsc_v[0];
                    let mut ysh = 0i32;
                    if use_el
                        && not_impossible(endsc_v)
                        && (endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1
                                    - state_delta(stt)) as f32))
                            > val
                    {
                        val = endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1 - state_delta(stt))
                                    as f32);
                        ysh = USED_EL;
                    }
                    for yoffset in 1..cnum {
                        let sc = a[y + yoffset].as_ref().unwrap()[jp][ip + 1] + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    val += sing_sc(esc_v, dsq[i as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    a[v].as_mut().unwrap()[jp][ip] = val;
                    if want_shadow {
                        shadow[v].as_mut().unwrap()[jp][ip] = ysh;
                    }
                }
            }
        } else if stt == MR_ST || stt == IR_ST {
            for ip in (0..=ipmax).rev() {
                a[v].as_mut().unwrap()[0][ip] = IMPOSSIBLE;
            }
            for jp in 1..=jpmax {
                let j = jp as i32 + j1;
                for ip in (0..=ipmax).rev() {
                    let y = cfirst;
                    let mut val = a[y].as_ref().unwrap()[jp - 1][ip] + tsc_v[0];
                    let mut ysh = 0i32;
                    if use_el
                        && not_impossible(endsc_v)
                        && (endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1
                                    - state_delta(stt)) as f32))
                            > val
                    {
                        val = endsc_v
                            + cm.el_selfsc
                                * (((jp as i32 + j1) - (ip as i32 + i0) + 1 - state_delta(stt))
                                    as f32);
                        ysh = USED_EL;
                    }
                    for yoffset in 1..cnum {
                        let sc = a[y + yoffset].as_ref().unwrap()[jp - 1][ip] + tsc_v[yoffset];
                        if sc > val {
                            val = sc;
                            ysh = yoffset as i32;
                        }
                    }
                    val += sing_sc(esc_v, dsq[j as usize]);
                    if val < IMPOSSIBLE {
                        val = IMPOSSIBLE;
                    }
                    a[v].as_mut().unwrap()[jp][ip] = val;
                    if want_shadow {
                        shadow[v].as_mut().unwrap()[jp][ip] = ysh;
                    }
                }
            }
        }

        // local begin bookkeeping (C:2418)
        if allow_begin {
            let av = a[v].as_ref().unwrap()[jpmax][0] + cm.beginsc[v];
            if av > bsc {
                b = v as i32;
                bsc = av;
            }
        }
        if allow_begin && v == 0 && bsc > a[0].as_ref().unwrap()[jpmax][0] {
            a[0].as_mut().unwrap()[jpmax][0] = bsc;
            if want_shadow {
                shadow[v].as_mut().unwrap()[jpmax][0] = USED_LOCAL_BEGIN;
            }
        }
    }

    if want_shadow {
        (Some(shadow), b, bsc)
    } else {
        (None, b, bsc)
    }
}

// ================================================================
// voutside() — C cm_dpsmall.c:2480 (vji coords)
// ================================================================
#[allow(clippy::too_many_arguments)]
fn voutside(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    r: usize,
    z: usize,
    i0: i32,
    i1: i32,
    j1: i32,
    j0: i32,
    use_el: bool,
    beta: &mut Vec<Option<VjiDeck>>,
) {
    let m = cm.m as usize;
    let local_end = cm.flags & CMH_LOCAL_END != 0;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;
    let jpmax = (j0 - j1) as usize;
    let ipmax = (i1 - i0) as usize;

    // Init root deck (C:2508)
    beta[r] = Some(new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE));
    beta[r].as_mut().unwrap()[jpmax][0] = 0.0;

    // EL deck + r->EL unroll (C:2521)
    if use_el && local_end {
        let mut eld = new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE);
        if not_impossible(cm.endsc[r]) {
            let esc_r = &cm.esc[r];
            match cm.sttype[r] as i32 {
                x if x == MP_ST => {
                    if !(i0 == i1 || j1 == j0) {
                        let escore = pair_sc(esc_r, dsq[i0 as usize], dsq[j0 as usize]);
                        eld[(j0 - j1 - 1) as usize][1] = cm.endsc[r]
                            + cm.el_selfsc * (((j0 - 1) - (i0 + 1) + 1) as f32)
                            + escore;
                    }
                }
                x if x == ML_ST || x == IL_ST => {
                    if i0 != i1 {
                        let escore = sing_sc(esc_r, dsq[i0 as usize]);
                        eld[(j0 - j1) as usize][1] =
                            cm.endsc[r] + cm.el_selfsc * ((j0 - (i0 + 1) + 1) as f32) + escore;
                    }
                }
                x if x == MR_ST || x == IR_ST => {
                    if j0 != j1 {
                        let escore = sing_sc(esc_r, dsq[j0 as usize]);
                        eld[(j0 - j1 - 1) as usize][0] =
                            cm.endsc[r] + cm.el_selfsc * (((j0 - 1) - i0 + 1) as f32) + escore;
                    }
                }
                x if x == S_ST || x == D_ST => {
                    eld[(j0 - j1) as usize][0] =
                        cm.endsc[r] + cm.el_selfsc * ((j0 - i0 + 1) as f32);
                }
                _ => {}
            }
        }
        beta[m] = Some(eld);
    }

    // Main loop (C:2585)
    for v in (r + 1)..=z {
        let mut deck = new_vji_deck(i0, i1, j1, j0, IMPOSSIBLE);
        if r == 0 && i0 == 1 && j0 == l && local_begin && cm.beginsc[v] > deck[jpmax][0] {
            deck[jpmax][0] = cm.beginsc[v];
        }
        let plast = cm.plast[v];
        let pnum = cm.pnum[v];
        for jp in (0..=jpmax).rev() {
            let j = jp as i32 + j1;
            for ip in 0..=ipmax {
                let i = ip as i32 + i0;
                let mut best = deck[jp][ip];
                let mut y = plast;
                while y > plast - pnum {
                    let yu = y as usize;
                    if (yu as i32) < r as i32 {
                        y -= 1;
                        continue;
                    }
                    let voffset = (v as i32 - cm.cfirst[yu]) as usize;
                    let yst = cm.sttype[yu] as i32;
                    let esc_y = &cm.esc[yu];
                    let tsc_y = &cm.tsc[yu];
                    let is_self = yu == v;
                    match yst {
                        x if x == MP_ST => {
                            if !(j == j0 || i == i0) {
                                let escore = pair_sc(esc_y, dsq[(i - 1) as usize], dsq[(j + 1) as usize]);
                                let pv = if is_self {
                                    deck[jp + 1][ip - 1]
                                } else {
                                    beta[yu].as_ref().unwrap()[jp + 1][ip - 1]
                                };
                                let sc = pv + tsc_y[voffset] + escore;
                                if sc > best {
                                    best = sc;
                                }
                            }
                        }
                        x if x == ML_ST || x == IL_ST => {
                            if i != i0 {
                                let escore = sing_sc(esc_y, dsq[(i - 1) as usize]);
                                let pv = if is_self {
                                    deck[jp][ip - 1]
                                } else {
                                    beta[yu].as_ref().unwrap()[jp][ip - 1]
                                };
                                let sc = pv + tsc_y[voffset] + escore;
                                if sc > best {
                                    best = sc;
                                }
                            }
                        }
                        x if x == MR_ST || x == IR_ST => {
                            if j != j0 {
                                let escore = sing_sc(esc_y, dsq[(j + 1) as usize]);
                                let pv = if is_self {
                                    deck[jp + 1][ip]
                                } else {
                                    beta[yu].as_ref().unwrap()[jp + 1][ip]
                                };
                                let sc = pv + tsc_y[voffset] + escore;
                                if sc > best {
                                    best = sc;
                                }
                            }
                        }
                        _ => {
                            let pv = if is_self {
                                deck[jp][ip]
                            } else {
                                beta[yu].as_ref().unwrap()[jp][ip]
                            };
                            let sc = pv + tsc_y[voffset];
                            if sc > best {
                                best = sc;
                            }
                        }
                    }
                    y -= 1;
                }
                if best < IMPOSSIBLE {
                    best = IMPOSSIBLE;
                }
                deck[jp][ip] = best;
            }
        }
        beta[v] = Some(deck);

        // v -> EL (C:2688)
        if use_el && not_impossible(cm.endsc[v]) {
            let vst = cm.sttype[v] as i32;
            let esc_v = &cm.esc[v];
            let endsc_v = cm.endsc[v];
            for jp in (0..=jpmax).rev() {
                let j = jp as i32 + j1;
                for ip in 0..=ipmax {
                    let i = ip as i32 + i0;
                    let cand = match vst {
                        x if x == MP_ST => {
                            if j == j0 || i == i0 {
                                None
                            } else {
                                let escore =
                                    pair_sc(esc_v, dsq[(i - 1) as usize], dsq[(j + 1) as usize]);
                                Some(
                                    beta[v].as_ref().unwrap()[jp + 1][ip - 1]
                                        + endsc_v
                                        + cm.el_selfsc * ((j - i + 1) as f32)
                                        + escore,
                                )
                            }
                        }
                        x if x == ML_ST || x == IL_ST => {
                            if i == i0 {
                                None
                            } else {
                                let escore = sing_sc(esc_v, dsq[(i - 1) as usize]);
                                Some(
                                    beta[v].as_ref().unwrap()[jp][ip - 1]
                                        + endsc_v
                                        + cm.el_selfsc * ((j - i + 1) as f32)
                                        + escore,
                                )
                            }
                        }
                        x if x == MR_ST || x == IR_ST => {
                            if j == j0 {
                                None
                            } else {
                                let escore = sing_sc(esc_v, dsq[(j + 1) as usize]);
                                Some(
                                    beta[v].as_ref().unwrap()[jp + 1][ip]
                                        + endsc_v
                                        + cm.el_selfsc * ((j - i + 1) as f32)
                                        + escore,
                                )
                            }
                        }
                        _ => Some(
                            beta[v].as_ref().unwrap()[jp][ip]
                                + endsc_v
                                + cm.el_selfsc * ((j - i + 1) as f32),
                        ),
                    };
                    if let Some(sc) = cand {
                        if sc > beta[m].as_ref().unwrap()[jp][ip] {
                            beta[m].as_mut().unwrap()[jp][ip] = sc;
                        }
                    }
                }
            }
        }
    }
}

// ================================================================
// inside_t() — C cm_dpsmall.c:2806 : run inside w/ shadow, trace back.
// ================================================================
#[allow(clippy::too_many_arguments)]
fn inside_t(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    j0: i32,
    allow_begin: bool,
) -> f32 {
    let m = cm.m as usize;
    let mut alpha: Vec<Option<VjdDeck>> = (0..=m).map(|_| None).collect();
    let (shadow_opt, b, _bsc) =
        inside(cm, dsq, l, r, z, i0, j0, &mut alpha, true, allow_begin);
    let shadow = shadow_opt.unwrap();
    let sc = alpha[r].as_ref().unwrap()[j0 as usize][(j0 - i0 + 1) as usize];

    // stack of (j, k, bifparent) triples
    let mut pda: Vec<i32> = Vec::new();
    let mut v = r as i32;
    let mut j = j0;
    let mut i = i0;
    let mut d = j0 - i0 + 1;

    loop {
        let vu = v as usize;
        let stt = cm.sttype[vu] as i32;
        if stt == B_ST {
            let k = shadow[vu].as_ref().unwrap()[j as usize][d as usize];
            pda.push(j);
            pda.push(k);
            pda.push(tr.n - 1);
            j -= k;
            d -= k;
            i = j - d + 1;
            let y = cm.cfirst[vu];
            insert_trace_node(tr, tr.n - 1, true, i, j, y);
            v = y;
        } else if stt == E_ST || cm.sttype[vu] as i32 == crate::constants::EL_ST {
            match pda.pop() {
                None => break,
                Some(bifparent) => {
                    let k = pda.pop().unwrap();
                    let jj = pda.pop().unwrap();
                    d = k;
                    j = jj;
                    let bstate = tr.state[bifparent as usize] as usize;
                    let y = cm.cnum[bstate];
                    i = j - d + 1;
                    insert_trace_node(tr, bifparent, false, i, j, y);
                    v = y;
                }
            }
        } else {
            let yoffset = shadow[vu].as_ref().unwrap()[j as usize][d as usize];
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
                _ => {}
            }
            d = j - i + 1;

            if yoffset == USED_EL {
                insert_trace_node(tr, tr.n - 1, true, i, j, cm.m);
                v = cm.m;
            } else if yoffset == USED_LOCAL_BEGIN {
                insert_trace_node(tr, tr.n - 1, true, i, j, b);
                v = b;
            } else {
                let y = cm.cfirst[vu] + yoffset;
                insert_trace_node(tr, tr.n - 1, true, i, j, y);
                v = y;
            }
        }
    }
    sc
}

// ================================================================
// vinside_t() — C cm_dpsmall.c:2943 : run vinside w/ shadow, trace back.
// ================================================================
#[allow(clippy::too_many_arguments)]
fn vinside_t(
    cm: &CM,
    dsq: &[u8],
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    i1: i32,
    j1: i32,
    j0: i32,
    use_el: bool,
    allow_begin: bool,
) -> f32 {
    // trivial case (C:2977)
    if r == z {
        insert_trace_node(tr, tr.n - 1, true, i0, j0, r as i32);
        return 0.0;
    }
    let m = cm.m as usize;
    let mut a: Vec<Option<VjiDeck>> = (0..=m).map(|_| None).collect();
    let (shadow_opt, b, _bsc) =
        vinside(cm, dsq, r, z, i0, i1, j1, j0, use_el, &mut a, true, allow_begin);
    let shadow = shadow_opt.unwrap();
    let sc = a[r].as_ref().unwrap()[(j0 - j1) as usize][0];

    let mut v = r as i32;
    let mut j = j0;
    let mut i = i0;
    loop {
        let jp = (j - j1) as usize;
        let ip = (i - i0) as usize;
        let vu = v as usize;
        let yoffset = shadow[vu].as_ref().unwrap()[jp][ip];
        match cm.sttype[vu] as i32 {
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
            _ => {}
        }

        if yoffset == USED_EL {
            insert_trace_node(tr, tr.n - 1, true, i, j, cm.m);
            break;
        } else if yoffset == USED_LOCAL_BEGIN {
            insert_trace_node(tr, tr.n - 1, true, i, j, b);
            v = b;
            if !use_el && v == z as i32 {
                break;
            }
        } else {
            let y = cm.cfirst[vu] + yoffset;
            insert_trace_node(tr, tr.n - 1, true, i, j, y);
            v = y;
            if !use_el && v == z as i32 {
                break;
            }
        }
    }
    sc
}

// ================================================================
// generic_splitter() — C cm_dpsmall.c:686
// ================================================================
fn generic_splitter(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    j0: i32,
) -> f32 {
    let m = cm.m as usize;
    // 1. RAMLIMIT==0 => never solve directly here; only via wedge boundary. But the
    //    "small enough" test is always false, so we always proceed to split.

    // 2. Traverse down from r, find first bifurcation.
    let mut v = r;
    while v as i32 <= z as i32 - 5 {
        if cm.sttype[v] as i32 == B_ST {
            break;
        }
        v += 1;
    }

    // 3. No bifurcation => wedge problem.
    if v as i32 > z as i32 - 5 {
        return wedge_splitter(cm, dsq, l, tr, r, z, i0, j0);
    }

    // Set up quartet r, v, w, y.
    let wst = cm.cfirst[v] as usize; // left S
    let yst = cm.cnum[v] as usize; // right S
    let (wend, yend) = if wst < yst {
        (yst - 1, z)
    } else {
        (z, wst - 1)
    };

    // shared matrix: inside fills w..z (alpha), outside fills r..v + M (beta).
    let mut mx: Vec<Option<VjdDeck>> = (0..=m).map(|_| None).collect();
    let (_, b1, b1_sc) = inside(cm, dsq, l, wst, wend, i0, j0, &mut mx, false, r == 0);
    let (_, b2, b2_sc) = inside(cm, dsq, l, yst, yend, i0, j0, &mut mx, false, r == 0);
    outside(cm, dsq, l, r, v, i0, j0, &mut mx);

    // Find optimal split at B.
    let bigw = j0 - i0 + 1;
    let mut best_sc = IMPOSSIBLE;
    let mut best_k = 0i32;
    let mut best_j = 0i32;
    let mut best_d = 0i32;
    for jp in 0..=bigw {
        let j = i0 - 1 + jp;
        for d in 0..=jp {
            for k in 0..=d {
                let sc = mx[wst].as_ref().unwrap()[(j - k) as usize][(d - k) as usize]
                    + mx[yst].as_ref().unwrap()[j as usize][k as usize]
                    + mx[v].as_ref().unwrap()[j as usize][d as usize];
                if sc > best_sc {
                    best_sc = sc;
                    best_k = k;
                    best_j = j;
                    best_d = d;
                }
            }
        }
    }

    // Local end: maybe better in EL?
    if cm.flags & CMH_LOCAL_END != 0 {
        for jp in 0..=bigw {
            let j = i0 - 1 + jp;
            for d in (0..=jp).rev() {
                let sc = mx[m].as_ref().unwrap()[j as usize][d as usize];
                if sc > best_sc {
                    best_sc = sc;
                    best_k = -1;
                    best_j = j;
                    best_d = d;
                }
            }
        }
    }

    // Local begin: maybe better in ROOT?
    if r == 0 && cm.flags & CMH_LOCAL_BEGIN != 0 {
        if b1_sc > best_sc {
            best_sc = b1_sc;
            best_k = -2;
            best_j = j0;
            best_d = bigw;
        }
        if b2_sc > best_sc {
            best_sc = b2_sc;
            best_k = -3;
            best_j = j0;
            best_d = bigw;
        }
    }

    // dispatch
    if best_k == -1 {
        v_splitter(cm, dsq, l, tr, r, v, i0, best_j - best_d + 1, best_j, j0, true);
        return best_sc;
    }
    if best_k == -2 {
        insert_trace_node(tr, tr.n - 1, true, i0, j0, b1);
        let z2 = subtree_find_end(cm, b1 as usize);
        generic_splitter(cm, dsq, l, tr, b1 as usize, z2, i0, j0);
        return best_sc;
    }
    if best_k == -3 {
        insert_trace_node(tr, tr.n - 1, true, i0, j0, b2);
        let z2 = subtree_find_end(cm, b2 as usize);
        generic_splitter(cm, dsq, l, tr, b2 as usize, z2, i0, j0);
        return best_sc;
    }

    // usual case: V problem + two generic problems
    v_splitter(cm, dsq, l, tr, r, v, i0, best_j - best_d + 1, best_j, j0, false);
    let tv = tr.n - 1;
    insert_trace_node(tr, tv, true, best_j - best_d + 1, best_j - best_k, wst as i32);
    generic_splitter(cm, dsq, l, tr, wst, wend, best_j - best_d + 1, best_j - best_k);
    insert_trace_node(tr, tv, false, best_j - best_k + 1, best_j, yst as i32);
    generic_splitter(cm, dsq, l, tr, yst, yend, best_j - best_k + 1, best_j);

    best_sc
}

// ================================================================
// wedge_splitter() — C cm_dpsmall.c:907
// ================================================================
fn wedge_splitter(
    cm: &CM,
    dsq: &[u8],
    l: i32,
    tr: &mut Parsetree,
    r: usize,
    z: usize,
    i0: i32,
    j0: i32,
) -> f32 {
    let m = cm.m as usize;
    // 1. Boundary condition (adjacent nodes) or RAMLIMIT(=0) => insideT.
    if cm.ndidx[z] == cm.ndidx[r] + 1 {
        return inside_t(cm, dsq, l, tr, r, z, i0, j0, r == 0);
    }

    // 2. split set w..y = midnode's split set.
    let midnode = cm.ndidx[r] + (cm.ndidx[z] - cm.ndidx[r]) / 2;
    let w = cm.nodemap[midnode as usize] as usize;
    let y = (cm.cfirst[w] - 1) as usize;

    // 3. inside up to w, outside down to y (separate matrices).
    let mut alpha: Vec<Option<VjdDeck>> = (0..=m).map(|_| None).collect();
    let (_, b, bsc) = inside(cm, dsq, l, w, z, i0, j0, &mut alpha, false, r == 0);
    let mut beta: Vec<Option<VjdDeck>> = (0..=m).map(|_| None).collect();
    outside(cm, dsq, l, r, y, i0, j0, &mut beta);

    // 4. optimal split.
    let bigw = j0 - i0 + 1;
    let mut best_sc = IMPOSSIBLE;
    let mut best_v: i32 = 0;
    let mut best_d = 0i32;
    let mut best_j = 0i32;
    for vv in w..=y {
        for jp in 0..=bigw {
            let j = i0 - 1 + jp;
            for d in 0..=jp {
                let sc = alpha[vv].as_ref().unwrap()[j as usize][d as usize]
                    + beta[vv].as_ref().unwrap()[j as usize][d as usize];
                if sc > best_sc {
                    best_sc = sc;
                    best_v = vv as i32;
                    best_d = d;
                    best_j = j;
                }
            }
        }
    }
    if cm.flags & CMH_LOCAL_END != 0 {
        for jp in 0..=bigw {
            let j = i0 - 1 + jp;
            for d in 0..=jp {
                let sc = beta[m].as_ref().unwrap()[j as usize][d as usize];
                if sc > best_sc {
                    best_sc = sc;
                    best_v = -1;
                    best_j = j;
                    best_d = d;
                }
            }
        }
    }
    if r == 0 && cm.flags & CMH_LOCAL_BEGIN != 0 && bsc > best_sc {
        best_sc = bsc;
        best_v = -2;
        best_j = j0;
        best_d = bigw;
    }

    if best_v == -1 {
        v_splitter(cm, dsq, l, tr, r, w, i0, best_j - best_d + 1, best_j, j0, true);
        return best_sc;
    }
    if best_v == -2 {
        insert_trace_node(tr, tr.n - 1, true, i0, j0, b);
        wedge_splitter(cm, dsq, l, tr, b as usize, z, i0, j0);
        return best_sc;
    }

    v_splitter(
        cm, dsq, l, tr, r, best_v as usize, i0, best_j - best_d + 1, best_j, j0, false,
    );
    wedge_splitter(cm, dsq, l, tr, best_v as usize, z, best_j - best_d + 1, best_j);
    best_sc
}

// ================================================================
// v_splitter() — C cm_dpsmall.c:1086
// ================================================================
#[allow(clippy::too_many_arguments)]
fn v_splitter(
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
) {
    let m = cm.m as usize;
    // 1. boundary condition or RAMLIMIT(=0) => vinsideT.
    if cm.ndidx[z] == cm.ndidx[r] + 1 || r == z {
        vinside_t(cm, dsq, tr, r, z, i0, i1, j1, j0, use_el, r == 0);
        return;
    }

    // 2. split set.
    let midnode = cm.ndidx[r] + (cm.ndidx[z] - cm.ndidx[r]) / 2;
    let w = cm.nodemap[midnode as usize] as usize;
    let y = (cm.cfirst[w] - 1) as usize;

    // 3. vinside up to w, voutside down to y.
    let mut alpha: Vec<Option<VjiDeck>> = (0..=m).map(|_| None).collect();
    let (_, b, bsc) = vinside(
        cm, dsq, w, z, i0, i1, j1, j0, use_el, &mut alpha, false, r == 0,
    );
    let mut beta: Vec<Option<VjiDeck>> = (0..=m).map(|_| None).collect();
    voutside(cm, dsq, l, r, y, i0, i1, j1, j0, use_el, &mut beta);

    // 4. optimal split.
    let mut best_sc = IMPOSSIBLE;
    let mut best_v: i32 = 0;
    let mut best_i = 0i32;
    let mut best_j = 0i32;
    for vv in w..=y {
        for ip in 0..=(i1 - i0) {
            for jp in 0..=(j0 - j1) {
                let sc = alpha[vv].as_ref().unwrap()[jp as usize][ip as usize]
                    + beta[vv].as_ref().unwrap()[jp as usize][ip as usize];
                if sc > best_sc {
                    best_sc = sc;
                    best_v = vv as i32;
                    best_i = ip + i0;
                    best_j = jp + j1;
                }
            }
        }
    }
    if use_el && cm.flags & CMH_LOCAL_END != 0 {
        for ip in 0..=(i1 - i0) {
            for jp in 0..=(j0 - j1) {
                let sc = beta[m].as_ref().unwrap()[jp as usize][ip as usize];
                if sc > best_sc {
                    best_sc = sc;
                    best_v = -1;
                    best_i = ip + i0;
                    best_j = jp + j1;
                }
            }
        }
    }
    if r == 0 && cm.flags & CMH_LOCAL_BEGIN != 0 && bsc > best_sc {
        best_sc = bsc;
        best_v = -2;
        best_i = i0;
        best_j = j0;
    }

    if best_v == -1 {
        v_splitter(cm, dsq, l, tr, r, w, i0, best_i, best_j, j0, true);
        return;
    }
    if best_v == -2 {
        if b != z as i32 {
            insert_trace_node(tr, tr.n - 1, true, i0, j0, b);
        }
        v_splitter(cm, dsq, l, tr, b as usize, z, i0, i1, j1, j0, use_el);
        return;
    }

    v_splitter(
        cm, dsq, l, tr, r, best_v as usize, i0, best_i, best_j, j0, false,
    );
    v_splitter(
        cm, dsq, l, tr, best_v as usize, z, best_i, i1, j1, best_j, use_el,
    );
}

// ================================================================
// CYKDivideAndConquer() — C cm_dpsmall.c:302 (top-level entry, r=0,i0=1,j0=L).
// ================================================================
/// Faithful port of C `CYKDivideAndConquer(cm, dsq, L, 0, 1, L, &tr, NULL, NULL)`.
/// Returns the optimal parsetree and its CYK bit score. `dsq` is 1..=L (dsq[0]
/// is a sentinel, unused).
pub fn cyk_divide_and_conquer(cm: &CM, dsq: &[u8], l: i32) -> (Parsetree, f32) {
    let mut tr = Parsetree::new(100);
    // init: attach the root S (state 0) at 1..L (C:324)
    insert_trace_node(&mut tr, -1, true, 1, l, ROOT_S);
    let z = cm.m - 1;
    // r == 0 for a full alignment; the r!=0 local-entry branch never applies here.
    let sc = generic_splitter(cm, dsq, l, &mut tr, 0, z as usize, 1, l);
    (tr, sc)
}
