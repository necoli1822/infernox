//! cm_alidisplay — faithful port of C Infernal 1.1.5 per-hit alignment display
//! for the `-g --nohmm --toponly --notrunc` global CM-search path.
//!
//! Three pieces, in dependency order (all J-mode / non-truncated / non-local,
//! which is all the `-g --nohmm` pipeline ever produces):
//!   1. [`CmConsensus`] — port of `display.c:CreateCMConsensus()` (+ its
//!      `createMultifurcationOrderChart` / `createFaceCharts` helpers). Builds the
//!      consensus display strings `cseq`/`cstr` and the node->consensus maps
//!      `lpos`/`rpos` (0-based, exactly as C's CMConsensus_t).
//!   2. [`cyk_align_global`] — global (non-banded) CYK alignment with a shadow
//!      matrix + traceback, a port of `cm_dpsmall.c:inside()` + `insideT()`
//!      (the `dmin==dmax==NULL`, `r==0`, global-config path). Produces a
//!      [`Parsetree`]. For a full-length global hit this yields the same optimal
//!      parse as C's QDB D&C CYK (`CYKDivideAndConquer`), since the optimum lies
//!      within the bands.
//!   3. [`cm_alidisplay_create`] — port of `cm_alidisplay.c:cm_alidisplay_Create()`
//!      main PDA loop (J-mode only), plus `ParsetreeToCMBounds()` (J-mode) for
//!      cfrom_emit/cto_emit. Emits the six display lines: ncline, csline,
//!      model(consensus), mline, aseq, rfline.

use crate::cm::CM;
use crate::constants::{
    B_ST, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST,
    BIF_ND, END_ND, MATL_ND, MATP_ND, MATR_ND, ROOT_ND, BEGL_ND, BEGR_ND,
    MATP_MP, MATP_ML, MATP_MR, MATL_ML, MATR_MR,
};
use crate::legacy::parsetree::Parsetree;

const IMPOSSIBLE: f32 = -1.0e36;
const USED_EL: i32 = -2;
const USED_LOCAL_BEGIN: i32 = -3;
const K: usize = 4; // RNA alphabet size (canonical)

/// RNA symbol for a digital code (0..3 -> A,C,G,U). Degenerate codes map to 'N'
/// (never occurs in the pure-ACGU validation regions, but kept safe).
#[inline]
fn sym(code: u8) -> u8 {
    match code {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'U',
        _ => b'N',
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
fn state_left_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST || x == ML_ST || x == IL_ST => 1,
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

// ===========================================================================
// 1. CmConsensus — port of CreateCMConsensus (display.c:652)
// ===========================================================================

/// Consensus display info for a CM (C `CMConsensus_t`). `cseq`/`cstr` are 0-based
/// [0..clen-1]; `lpos`/`rpos` are 0-based node->consensus maps [0..nodes-1] (same
/// convention as C, so `cm.consensus[lpos[nd]+1]` is the 1-based model residue).
#[derive(Debug, Clone)]
pub struct CmConsensus {
    pub cseq: Vec<u8>,
    pub cstr: Vec<u8>,
    pub lpos: Vec<i32>,
    pub rpos: Vec<i32>,
    pub clen: i32,
}

/// C `createMultifurcationOrderChart` (display.c:882). height[nd] = multifurcation
/// order beneath node nd's master subtree.
fn create_multiorder(cm: &CM) -> Vec<i32> {
    let nodes = cm.nodes as usize;
    let mut height = vec![0i32; nodes];
    let mut seg_has_pairs = vec![0i32; nodes];
    for nd in (0..nodes).rev() {
        let v = cm.nodemap[nd] as usize;
        let stid = cm.stid[v] as i32;
        if stid == MATP_MP {
            seg_has_pairs[nd] = 1;
        } else if stid == crate::constants::END_E {
            seg_has_pairs[nd] = 0;
        } else if stid == crate::constants::BIF_B {
            seg_has_pairs[nd] = 0;
        } else {
            seg_has_pairs[nd] = seg_has_pairs[nd + 1];
        }

        if stid == crate::constants::END_E {
            height[nd] = 0;
        } else if stid == crate::constants::BIF_B {
            let left = cm.ndidx[cm.cfirst[v] as usize] as usize;
            let right = cm.ndidx[cm.cnum[v] as usize] as usize;
            height[nd] = (height[left] + seg_has_pairs[left]).max(height[right] + seg_has_pairs[right]);
        } else {
            height[nd] = height[nd + 1];
        }
    }
    height
}

/// C `createFaceCharts` (display.c:952). Returns (inface, outface). Faithfully
/// reproduces the known BEGR `y`-carry quirk (see the C comment "THIS FUNCTION IS
/// BUGGY"): `y` persists across the ascending outface loop, set at each BEGL node.
fn create_facecharts(cm: &CM) -> (Vec<i32>, Vec<i32>) {
    let nodes = cm.nodes as usize;
    let mut inface = vec![0i32; nodes];
    let mut outface = vec![0i32; nodes];

    for nd in (0..nodes).rev() {
        let v = cm.nodemap[nd] as usize;
        let ndt = cm.ndtype[nd] as i32;
        if ndt == END_ND {
            inface[nd] = 0;
        } else if ndt == BIF_ND {
            let left = cm.ndidx[cm.cfirst[v] as usize] as usize;
            let right = cm.ndidx[cm.cnum[v] as usize] as usize;
            inface[nd] = inface[left] + inface[right];
        } else if cm.ndtype[nd + 1] as i32 == MATP_ND {
            inface[nd] = 1;
        } else {
            inface[nd] = inface[nd + 1];
        }
    }

    let mut y: usize = 0; // carried across iterations (matches C's buggy scope)
    for nd in 0..nodes {
        let v = cm.nodemap[nd] as usize;
        let ndt = cm.ndtype[nd] as i32;
        if ndt == ROOT_ND {
            outface[nd] = 0;
        } else if ndt == BEGL_ND {
            let parent = cm.ndidx[cm.plast[v] as usize] as usize;
            y = cm.nodemap[parent] as usize;
            let right = cm.ndidx[cm.cnum[y] as usize] as usize;
            outface[nd] = outface[parent] + inface[right];
        } else if ndt == BEGR_ND {
            let parent = cm.ndidx[cm.plast[v] as usize] as usize;
            let left = cm.ndidx[cm.cfirst[y] as usize] as usize;
            outface[nd] = outface[parent] + inface[left];
        } else {
            let parent = nd - 1;
            if cm.ndtype[parent] as i32 == MATP_ND {
                outface[nd] = 1;
            } else {
                outface[nd] = outface[parent];
            }
        }
    }
    (inface, outface)
}

#[inline]
fn f_argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// C `CreateCMConsensus` (display.c:652). Builds consensus display strings.
pub fn create_cm_consensus(cm: &CM) -> CmConsensus {
    let nodes = cm.nodes as usize;
    let mut lpos = vec![-1i32; nodes];
    let mut rpos = vec![-1i32; nodes];
    let mut cseq: Vec<u8> = Vec::new();
    let mut cstr: Vec<u8> = Vec::new();
    let pthresh: f32 = 3.0;
    let sthresh: f32 = 1.0;

    let multiorder = create_multiorder(cm);
    let (inface, outface) = create_facecharts(cm);

    // PDA items
    enum Item {
        State(i32),
        // (nd, pairpartner, rstruc, rchar)
        Residue(i32, i32, u8, u8),
        Marker(i32),
    }
    let mut pda: Vec<Item> = Vec::new();
    let mut cpos: i32 = 0;
    pda.push(Item::State(0));

    while let Some(item) = pda.pop() {
        match item {
            Item::Residue(nd, _pairpartner, rstruc, rchar) => {
                rpos[nd as usize] = cpos;
                cseq.push(rchar);
                cstr.push(rstruc);
                cpos += 1;
            }
            Item::Marker(nd) => {
                rpos[nd as usize] = cpos - 1;
            }
            Item::State(v) => {
                let vu = v as usize;
                let nd = cm.ndidx[vu] as usize;
                let stid = cm.stid[vu] as i32;
                let mut lchar: u8 = 0;
                let mut rchar: u8 = 0;
                let mut lstruc: u8 = 0;
                let mut rstruc: u8 = 0;

                if stid == MATP_MP {
                    let x = f_argmax(&cm.esc[vu][0..K * K]);
                    let mut lc = sym((x / K) as u8);
                    let mut rc = sym((x % K) as u8);
                    if cm.esc[vu][x] < pthresh {
                        lc = lc.to_ascii_lowercase();
                        rc = rc.to_ascii_lowercase();
                    }
                    lchar = lc;
                    rchar = rc;
                    match multiorder[nd] {
                        0 => { lstruc = b'<'; rstruc = b'>'; }
                        1 => { lstruc = b'('; rstruc = b')'; }
                        2 => { lstruc = b'['; rstruc = b']'; }
                        _ => { lstruc = b'{'; rstruc = b'}'; }
                    }
                } else if stid == MATL_ML {
                    let x = f_argmax(&cm.esc[vu][0..K]);
                    let mut lc = sym(x as u8);
                    if cm.esc[vu][x] < sthresh {
                        lc = lc.to_ascii_lowercase();
                    }
                    lchar = lc;
                    if outface[nd] == 0 {
                        lstruc = b':';
                    } else if inface[nd] == 0 && outface[nd] == 1 {
                        lstruc = b'_';
                    } else if inface[nd] == 1 && outface[nd] == 1 {
                        lstruc = b'-';
                    } else {
                        lstruc = b',';
                    }
                    rstruc = b' ';
                } else if stid == MATR_MR {
                    let x = f_argmax(&cm.esc[vu][0..K]);
                    let mut rc = sym(x as u8);
                    if cm.esc[vu][x] < sthresh {
                        rc = rc.to_ascii_lowercase();
                    }
                    rchar = rc;
                    if outface[nd] == 0 {
                        rstruc = b':';
                    } else if inface[nd] == 0 && outface[nd] == 1 {
                        rstruc = b'?';
                    } else if inface[nd] == 1 && outface[nd] == 1 {
                        rstruc = b'-';
                    } else {
                        rstruc = b',';
                    }
                    lstruc = b' ';
                }

                lpos[nd] = cpos;
                if lchar != 0 {
                    cseq.push(lchar);
                    cstr.push(lstruc);
                    cpos += 1;
                }
                if rchar != 0 {
                    let pairpartner = if lchar != 0 { cpos - 1 } else { -1 };
                    pda.push(Item::Residue(nd as i32, pairpartner, rstruc, rchar));
                } else {
                    pda.push(Item::Marker(nd as i32));
                }

                // Transit to consensus states only.
                let stt = cm.sttype[vu] as i32;
                if stt == B_ST {
                    // push right S then left S (left pops first)
                    pda.push(Item::State(cm.cnum[vu]));
                    pda.push(Item::State(cm.cfirst[vu]));
                } else if stt != E_ST {
                    let child = (cm.cfirst[vu] + cm.cnum[vu] - 1) as usize;
                    let nv = cm.nodemap[cm.ndidx[child] as usize];
                    pda.push(Item::State(nv));
                }
            }
        }
    }

    let clen = cpos;
    CmConsensus { cseq, cstr, lpos, rpos, clen }
}

// ===========================================================================
// 2. Global CYK alignment: inside() fill + insideT() traceback (non-banded)
// ===========================================================================

/// Global non-banded CYK alignment of `dsq[1..=lp]` to the (global-config) CM.
/// Port of `cm_dpsmall.c:inside()` (full matrix + shadow) followed by `insideT()`
/// traceback. Returns (parsetree, score). `dsq` must be indexed so dsq[1..=lp] are
/// the hit residues (dsq[0] unused).
pub fn cyk_align_global(cm: &CM, dsq: &[u8], lp: i32) -> (Parsetree, f32) {
    let m = cm.m as usize;
    let w = lp as usize; // subsequence length (i0=1, j0=lp)
    let stride = w + 1;
    let el = cm.el_selfsc;

    // alpha[v] : flat (w+1)*(w+1), index j*stride + d. yshadow/kshadow likewise.
    let mut alpha: Vec<Vec<f32>> = vec![Vec::new(); m];
    let mut yshadow: Vec<Vec<i32>> = vec![Vec::new(); m];
    // kshadow only for B states
    let mut kshadow: Vec<Vec<i32>> = vec![Vec::new(); m];

    // End deck (shared value): E states have alpha[j][0]=0, else IMPOSSIBLE.
    // We simply materialize per-E-state below.

    let mut b: i32 = -1;
    let mut bsc: f32 = IMPOSSIBLE;
    let allow_begin = true; // r==0; harmless in global (beginsc all IMPOSSIBLE)

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        if stt == E_ST {
            // end deck
            let mut a = vec![IMPOSSIBLE; stride * stride];
            for j in 0..=w {
                a[j * stride] = 0.0; // d=0
            }
            alpha[v] = a;
            continue;
        }
        let mut a = vec![IMPOSSIBLE; stride * stride];
        let mut ys = vec![USED_EL; stride * stride];
        let mut ks = vec![0i32; stride * stride];
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let tsc_v = &cm.tsc[v];
        let esc_v = &cm.esc[v];
        let endsc_v = cm.endsc[v];
        let sd = state_delta(stt);

        if stt == D_ST || stt == S_ST {
            for jp in 0..=w {
                let j = jp; // i0=1 => j = jp
                for d in 0..=jp {
                    let idx = j * stride + d;
                    let mut best = endsc_v + el * (d as i32 - sd) as f32;
                    let mut bsh = USED_EL;
                    for yo in 0..cnum {
                        let sc = alpha[cfirst + yo][idx] + tsc_v[yo];
                        if sc > best {
                            best = sc;
                            bsh = yo as i32;
                        }
                    }
                    if best < IMPOSSIBLE {
                        best = IMPOSSIBLE;
                    }
                    a[idx] = best;
                    ys[idx] = bsh;
                }
            }
        } else if stt == B_ST {
            let y = cfirst;
            let z = cm.cnum[v] as usize;
            for jp in 0..=w {
                let j = jp;
                for d in 0..=jp {
                    let idx = j * stride + d;
                    // k=0 term
                    let mut best = alpha[y][j * stride + d] + alpha[z][j * stride];
                    let mut bk = 0i32;
                    for k in 1..=d {
                        let sc = alpha[y][(j - k) * stride + (d - k)] + alpha[z][j * stride + k];
                        if sc > best {
                            best = sc;
                            bk = k as i32;
                        }
                    }
                    if best < IMPOSSIBLE {
                        best = IMPOSSIBLE;
                    }
                    a[idx] = best;
                    ks[idx] = bk;
                }
            }
        } else if stt == MP_ST {
            for jp in 0..=w {
                let j = jp;
                if jp >= 1 {
                    // d=0 and d=1 impossible (already IMPOSSIBLE)
                }
                for d in 2..=jp {
                    let idx = j * stride + d;
                    let mut best = endsc_v + el * (d as i32 - sd) as f32;
                    let mut bsh = USED_EL;
                    for yo in 0..cnum {
                        let sc = alpha[cfirst + yo][(j - 1) * stride + (d - 2)] + tsc_v[yo];
                        if sc > best {
                            best = sc;
                            bsh = yo as i32;
                        }
                    }
                    let i = j - d + 1; // 1-based
                    best += pair_score(esc_v, dsq[i], dsq[j]);
                    if best < IMPOSSIBLE {
                        best = IMPOSSIBLE;
                    }
                    a[idx] = best;
                    ys[idx] = bsh;
                }
            }
        } else if stt == IL_ST || stt == ML_ST {
            // IL self-loops (child == v): read the in-progress deck `a` for that case
            // (d-1 < d, already computed this same j-row).
            for jp in 0..=w {
                let j = jp;
                for d in 1..=jp {
                    let idx = j * stride + d;
                    let mut best = endsc_v + el * (d as i32 - sd) as f32;
                    let mut bsh = USED_EL;
                    let cidx = j * stride + (d - 1);
                    for yo in 0..cnum {
                        let child = cfirst + yo;
                        let cell = if child == v { a[cidx] } else { alpha[child][cidx] };
                        let sc = cell + tsc_v[yo];
                        if sc > best {
                            best = sc;
                            bsh = yo as i32;
                        }
                    }
                    let i = j - d + 1;
                    best += singlet_score(esc_v, dsq[i]);
                    if best < IMPOSSIBLE {
                        best = IMPOSSIBLE;
                    }
                    a[idx] = best;
                    ys[idx] = bsh;
                }
            }
        } else if stt == IR_ST || stt == MR_ST {
            // IR self-loops (child == v): read `a` at (j-1, d-1), computed in a prior
            // j-row iteration.
            for jp in 0..=w {
                let j = jp;
                for d in 1..=jp {
                    let idx = j * stride + d;
                    let mut best = endsc_v + el * (d as i32 - sd) as f32;
                    let mut bsh = USED_EL;
                    let cidx = (j - 1) * stride + (d - 1);
                    for yo in 0..cnum {
                        let child = cfirst + yo;
                        let cell = if child == v { a[cidx] } else { alpha[child][cidx] };
                        let sc = cell + tsc_v[yo];
                        if sc > best {
                            best = sc;
                            bsh = yo as i32;
                        }
                    }
                    best += singlet_score(esc_v, dsq[j]);
                    if best < IMPOSSIBLE {
                        best = IMPOSSIBLE;
                    }
                    a[idx] = best;
                    ys[idx] = bsh;
                }
            }
        }

        // local begin bookkeeping (harmless global: beginsc IMPOSSIBLE)
        let root_idx = w * stride + w;
        if allow_begin && a[root_idx] + cm.beginsc[v] > bsc {
            b = v as i32;
            bsc = a[root_idx] + cm.beginsc[v];
        }
        if allow_begin && v == 0 && bsc > a[root_idx] {
            a[root_idx] = bsc;
            ys[root_idx] = USED_LOCAL_BEGIN;
        }

        alpha[v] = a;
        yshadow[v] = ys;
        kshadow[v] = ks;
    }

    // ---- traceback (insideT) ----
    let mut tr = Parsetree::new(w + 4);
    // root
    tr.add_node(1, lp, 0, -1, -1, -1);
    // pda: (bifparent_index, saved_d, saved_j)
    let mut pda: Vec<(i32, i32, i32)> = Vec::new();
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = lp;
    let mut d: i32 = lp;
    loop {
        // E or EL: swing over to a pending right subtree, or finish. Checked first
        // so v==cm.m (EL) never indexes sttype (which is only 0..m-1).
        if v == cm.m || cm.sttype[v as usize] as i32 == E_ST {
            match pda.pop() {
                None => break,
                Some((bifparent, sd, sj)) => {
                    d = sd;
                    j = sj;
                    let bstate = tr.state[bifparent as usize];
                    let y = cm.cnum[bstate as usize];
                    i = j - d + 1;
                    let idx = tr.add_node(i, j, y, -1, -1, bifparent);
                    set_right_child(&mut tr, bifparent, idx);
                    v = y;
                    continue;
                }
            }
        }
        let vu = v as usize;
        let stt = cm.sttype[vu] as i32;
        if stt == B_ST {
            let k = kshadow[vu][(j as usize) * stride + d as usize];
            let parent = tr.n - 1;
            pda.push((parent, k, j));
            j -= k;
            d -= k;
            i = j - d + 1;
            let y = cm.cfirst[vu];
            let idx = tr.add_node(i, j, y, -1, -1, parent);
            set_left_child(&mut tr, parent, idx);
            v = y;
        } else {
            let yoffset = yshadow[vu][(j as usize) * stride + d as usize];
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
            set_left_child(&mut tr, parent, idx);
            v = ny;
        }
    }

    let sc = alpha[0][w * stride + w];
    (tr, sc)
}

#[inline]
fn set_left_child(tr: &mut Parsetree, parent: i32, child: i32) {
    if parent >= 0 {
        tr.nxtl[parent as usize] = child;
    }
}
#[inline]
fn set_right_child(tr: &mut Parsetree, parent: i32, child: i32) {
    if parent >= 0 {
        tr.nxtr[parent as usize] = child;
    }
}

/// Pair emission score esc[i*4+j] (C: canonical path of inside()'s MP block).
#[inline]
fn pair_score(esc: &[f32], di: u8, dj: u8) -> f32 {
    if (di as usize) < K && (dj as usize) < K {
        esc[di as usize * K + dj as usize]
    } else {
        // degenerate pair: uniform average over canonical (rare; not in test set)
        let mut s = 0.0f32;
        for a in 0..K {
            for c in 0..K {
                s += esc[a * K + c];
            }
        }
        s / (K * K) as f32
    }
}
/// Singlet emission score esc[i] (C: canonical path).
#[inline]
fn singlet_score(esc: &[f32], di: u8) -> f32 {
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

// ===========================================================================
// 3. cm_alidisplay_create — port of cm_alidisplay.c:cm_alidisplay_Create (J mode)
// ===========================================================================

/// The per-hit alignment display: the six annotation lines plus the model bounds.
/// Lines are full concatenated strings (C's `cm_alidisplay_Print` chunks them into
/// ~90-col blocks at output time). All J-mode / non-truncated.
#[derive(Debug, Clone)]
pub struct CmAliDisplay {
    /// aligned target sequence (uppercase=consensus, lowercase=insert, '-'=deletion)
    pub aseq: String,
    /// consensus secondary structure (`#=GC SS_cons`) line
    pub csline: String,
    /// noncanonical-mark line ('v'/'?'/' ')
    pub ncline: String,
    /// consensus/model residue line (cm.consensus when CMH_CONS, else cmcons.cseq)
    pub model: String,
    /// match/midline
    pub mline: String,
    /// reference-annotation (RF) line ('' if the model has no RF annotation)
    pub rfline: String,
    /// first model position spanned (C ad->cfrom_emit); tblout `mdl from`
    pub cfrom_emit: i32,
    /// final model position spanned (C ad->cto_emit); tblout `mdl to`
    pub cto_emit: i32,
    /// first model position of the (guessed) full parse (C ad->cfrom_span). Equals
    /// cfrom_emit for non-truncated (J-mode) parses.
    pub cfrom_span: i32,
    /// final model position of the (guessed) full parse (C ad->cto_span). Equals
    /// cto_emit for non-truncated parses.
    pub cto_span: i32,
    /// tblout `trunc` column: "no" / "5'" / "3'" / "5'&3'" (C cm_alidisplay_TruncString).
    pub trunc: String,
    /// pipeline pass index the hit was found in (tblout `pass` column).
    pub pass_idx: i32,
}

/// esl_abc_FAvgScore for a possibly-degenerate residue against a singlet esc[]
/// vector (canonical: esc[di]).
#[inline]
fn avg_singlet(esc: &[f32], di: u8) -> f32 {
    singlet_score(esc, di)
}

/// C `bp_is_canonical` (cm_alidisplay.c:1021).
#[inline]
fn bp_is_canonical(lseq: u8, rseq: u8) -> bool {
    let l = lseq.to_ascii_uppercase();
    let r = rseq.to_ascii_uppercase();
    matches!(
        (l, r),
        (b'A', b'U') | (b'A', b'T')
            | (b'C', b'G')
            | (b'G', b'C') | (b'G', b'U') | (b'G', b'T')
            | (b'U', b'A') | (b'U', b'G')
            | (b'T', b'A') | (b'T', b'G')
    )
}

enum PItem {
    State(i32),
    // (nnc, str, cons, mid, seq, rf)
    Residue(u8, u8, u8, u8, u8, u8),
}

/// Build the alignment display from a parsetree over the hit subsequence.
/// `dsq[1..=lp]` are the hit residues. `cmcons` is the model consensus. `cm` must
/// be the global-config CM the parsetree was aligned to.
pub fn cm_alidisplay_create(
    cm: &CM,
    cmcons: &CmConsensus,
    tr: &Parsetree,
    dsq: &[u8],
) -> CmAliDisplay {
    let use_cons = !cm.consensus.is_empty(); // CMH_CONS
    let use_rf = !cm.rf.is_empty(); // CMH_RF

    let mut aseq: Vec<u8> = Vec::new();
    let mut csline: Vec<u8> = Vec::new();
    let mut ncline: Vec<u8> = Vec::new();
    let mut model: Vec<u8> = Vec::new();
    let mut mline: Vec<u8> = Vec::new();
    let mut rfline: Vec<u8> = Vec::new();

    let mut pda: Vec<PItem> = Vec::new();
    pda.push(PItem::State(0));

    while let Some(item) = pda.pop() {
        match item {
            PItem::Residue(nnc, str_c, cons, mid, seq, rf) => {
                ncline.push(nnc);
                csline.push(str_c);
                model.push(cons);
                mline.push(mid);
                aseq.push(seq);
                if use_rf {
                    rfline.push(rf);
                }
            }
            PItem::State(ti) => {
                let ti = ti as usize;
                let v = tr.state[ti];
                if v == cm.m {
                    // EL local end — never occurs in global nohmm; ignore safely.
                    // (kept minimal; no residues emitted through EL here.)
                    continue;
                }
                let vu = v as usize;
                let stt = cm.sttype[vu] as i32;
                let stid = cm.stid[vu] as i32;
                let nd = cm.ndidx[vu] as usize;
                let ndt = cm.ndtype[nd] as i32;
                let lc = cmcons.lpos[nd]; // 0-based
                let rc = cmcons.rpos[nd];
                let emitl = tr.emitl[ti];
                let emitr = tr.emitr[ti];
                let symi = dsq[emitl as usize];
                let symj = dsq[emitr as usize];

                let mut do_left = false;
                let mut do_right = false;
                let mut lrf = b' ';
                let mut rrf = b' ';
                let mut lstr = b' ';
                let mut rstr = b' ';
                let mut lcons = b' ';
                let mut rcons = b' ';
                let mut lseq = b' ';
                let mut rseq = b' ';

                if stt == IL_ST {
                    do_left = true;
                    lrf = b'.';
                    lstr = b'.';
                    lcons = b'.';
                    lseq = sym(symi).to_ascii_lowercase();
                } else if stt == IR_ST {
                    do_right = true;
                    rrf = b'.';
                    rstr = b'.';
                    rcons = b'.';
                    rseq = sym(symj).to_ascii_lowercase();
                } else {
                    if ndt == MATP_ND || ndt == MATL_ND {
                        do_left = true;
                        if use_rf {
                            lrf = cm.rf[(lc + 1) as usize];
                        }
                        lstr = cmcons.cstr[lc as usize];
                        lcons = if use_cons {
                            cm.consensus[(lc + 1) as usize]
                        } else {
                            cmcons.cseq[lc as usize]
                        };
                        if stt == MP_ST || stt == ML_ST {
                            lseq = sym(symi);
                        } else {
                            lseq = b'-';
                        }
                    }
                    if ndt == MATP_ND || ndt == MATR_ND {
                        do_right = true;
                        if use_rf {
                            rrf = cm.rf[(rc + 1) as usize];
                        }
                        rstr = cmcons.cstr[rc as usize];
                        rcons = if use_cons {
                            cm.consensus[(rc + 1) as usize]
                        } else {
                            cmcons.cseq[rc as usize]
                        };
                        if stt == MP_ST || stt == MR_ST {
                            rseq = sym(symj);
                        } else {
                            rseq = b'-';
                        }
                    }
                }

                // mline (lmid/rmid) + ncline (lnnc/rnnc), all J mode
                let mut lmid = b' ';
                let mut rmid = b' ';
                let mut lnnc = b' ';
                let mut rnnc = b' ';
                if stt == MP_ST {
                    let tmpsc = pair_score(&cm.esc[vu], symi, symj);
                    if lseq == lcons.to_ascii_uppercase() && rseq == rcons.to_ascii_uppercase() {
                        lmid = lseq;
                        rmid = rseq;
                    } else if tmpsc >= 0.0 {
                        lmid = b':';
                        rmid = b':';
                    }
                    if tmpsc < 0.0 && !bp_is_canonical(lseq, rseq) {
                        lnnc = b'v';
                        rnnc = b'v';
                    }
                } else if (stt == ML_ST || stt == IL_ST) && do_left {
                    if lseq == lcons.to_ascii_uppercase() {
                        lmid = lseq;
                    } else if avg_singlet(&cm.esc[vu], symi) > 0.0 {
                        lmid = b'+';
                    }
                } else if (stt == MR_ST || stt == IR_ST) && do_right {
                    if rseq == rcons.to_ascii_uppercase() {
                        rmid = rseq;
                    } else if avg_singlet(&cm.esc[vu], symj) > 0.0 {
                        rmid = b'+';
                    }
                }
                if stid == MATP_ML || stid == MATP_MR {
                    lnnc = b'v';
                    rnnc = b'v';
                }

                if do_left {
                    ncline.push(lnnc);
                    csline.push(lstr);
                    model.push(lcons);
                    mline.push(lmid);
                    aseq.push(lseq);
                    if use_rf {
                        rfline.push(lrf);
                    }
                }
                if do_right {
                    pda.push(PItem::Residue(rnnc, rstr, rcons, rmid, rseq, rrf));
                }

                // children: right first, so left pops first
                if tr.nxtr[ti] != -1 {
                    pda.push(PItem::State(tr.nxtr[ti]));
                }
                if tr.nxtl[ti] != -1 {
                    pda.push(PItem::State(tr.nxtl[ti]));
                }
            }
        }
    }

    let (cfrom_emit, cto_emit) = parsetree_to_cm_bounds(cm, cmcons, tr);

    CmAliDisplay {
        aseq: String::from_utf8(aseq).unwrap(),
        csline: String::from_utf8(csline).unwrap(),
        ncline: String::from_utf8(ncline).unwrap(),
        model: String::from_utf8(model).unwrap(),
        mline: String::from_utf8(mline).unwrap(),
        rfline: String::from_utf8(rfline).unwrap(),
        cfrom_emit,
        cto_emit,
        // J-mode / non-truncated: span == emit, no truncation, standard pass.
        cfrom_span: cfrom_emit,
        cto_span: cto_emit,
        trunc: "no".to_string(),
        pass_idx: 1,
    }
}

/// C `ParsetreeToCMBounds` (cm_parsetree.c:2611), J-mode (PLI_PASS_STD_ANY) path.
/// Returns (cfrom_emit, cto_emit). Uses the CmConsensus lpos/rpos (== emit map).
fn parsetree_to_cm_bounds(cm: &CM, cmcons: &CmConsensus, tr: &Parsetree) -> (i32, i32) {
    let clen = cmcons.clen;
    let mut cfrom_emit = clen + 1;
    let mut cto_emit = 0;
    for ti in 0..tr.n as usize {
        let v = tr.state[ti];
        if v == cm.m {
            continue; // EL: J-mode nohmm never produces one
        }
        let vu = v as usize;
        let stt = cm.sttype[vu] as i32;
        let nd = cm.ndidx[vu] as usize;
        let ndt = cm.ndtype[nd] as i32;
        // emap-style lpos/rpos (1-based). cmcons.lpos is 0-based -> +1.
        let base_lpos = cmcons.lpos[nd] + 1;
        let base_rpos = cmcons.rpos[nd] + 1;
        let lpos = if ndt == MATP_ND || ndt == MATL_ND { base_lpos } else { base_lpos + 1 };
        let rpos = if ndt == MATP_ND || ndt == MATR_ND { base_rpos } else { base_rpos - 1 };

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
        if is_left {
            cfrom_emit = cfrom_emit.min(lpos + insert_sd);
            cto_emit = cto_emit.max(lpos);
        }
        if is_right {
            cfrom_emit = cfrom_emit.min(rpos + insert_sd);
            cto_emit = cto_emit.max(rpos);
        }
    }
    (cfrom_emit, cto_emit)
}
