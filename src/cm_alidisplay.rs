//! cm_alidisplay — faithful port of C Infernal 1.1.5 per-hit alignment display
//! (`cm_alidisplay.c`).
//!
//!   1. [`CmConsensus`] — port of `display.c:CreateCMConsensus()` (+ its
//!      `createMultifurcationOrderChart` / `createFaceCharts` helpers). Builds the
//!      consensus display strings `cseq`/`cstr` and the node->consensus maps
//!      `lpos`/`rpos` (0-based, exactly as C's CMConsensus_t).
//!   2. [`cyk_align_global`] — global (non-banded) CYK alignment with a shadow
//!      matrix + traceback, a port of `cm_dpsmall.c:inside()` + `insideT()`
//!      (the `dmin==dmax==NULL`, `r==0`, global-config path). Used by the
//!      `-g --nohmm`/`--max` paths.
//!   3. [`cm_alidisplay_create`] — J-mode-only [`CmAliDisplay`] builder (no PP),
//!      used by the `--nohmm`/`--max` paths.
//!   4. [`cm_alidisplay_create_full`] — FULL mode-aware port of
//!      `cm_alidisplay.c:cm_alidisplay_Create()`: the six display lines PLUS the
//!      `PP` posterior line and the 5'/3' truncated local-end markers
//!      (`<[n]*` / `*[n]>`), driven by each parse node's marginal mode (J/L/R/T).
//!      Used by the DEFAULT cmsearch pipeline (standard + truncated passes).
//!   5. [`cm_alidisplay_print`] — port of `cm_alidisplay.c:cm_alidisplay_Print()`:
//!      the block-wrapping output loop + local-end run parser.

use crate::cm::CM;
use crate::constants::{
    B_ST, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST,
    BIF_ND, END_ND, MATL_ND, MATP_ND, MATR_ND, ROOT_ND, BEGL_ND, BEGR_ND,
    MATP_MP, MATP_ML, MATP_MR, MATL_ML, MATR_MR,
};
use crate::parsetree::Parsetree;

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
    cyk_align_maybe_banded(cm, dsq, lp, None, None)
}

/// QDB-banded (or non-banded) global CYK alignment, a port of
/// `cm_dpsmall.c:inside()`/`inside_b()` + `insideT()` (the `r==0`, small-problem
/// base case of `CYKDivideAndConquer`/`generic_splitter[_b]`). When `dmin`/`dmax`
/// are `Some`, each state `v`'s `d` is confined to `[dmin[v], dmax[v]]` and a
/// bifurcation's split `k` to the child bands (C `inside_b`, cm_dpsmall.c:4469-4534);
/// the `--nohmm` path passes `cm.dmin2/dmax2`. When `None`, it is fully non-banded
/// (equivalent to `dmin=0, dmax=W` everywhere), the `--max` path. The traceback
/// (`insideT`) is band-agnostic — the shadow matrix already encodes banded choices.
pub fn cyk_align_maybe_banded(
    cm: &CM,
    dsq: &[u8],
    lp: i32,
    dmin: Option<&[i32]>,
    dmax: Option<&[i32]>,
) -> (Parsetree, f32) {
    let m = cm.m as usize;
    let w = lp as usize; // subsequence length (i0=1, j0=lp)
    let stride = w + 1;
    let el = cm.el_selfsc;
    // Per-state band [dmin,dmax]; non-banded => [0, W]. (C inside_b treats cells
    // outside the band as IMPOSSIBLE; our decks init to IMPOSSIBLE so only in-band
    // cells are ever written.)
    let band = |v: usize| -> (i32, i32) {
        match (dmin, dmax) {
            (Some(dl), Some(dh)) => (dl[v], dh[v]),
            _ => (0, w as i32),
        }
    };

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

        let (dmn, dmx) = band(v);
        if stt == D_ST || stt == S_ST {
            let dlo = dmn.max(0);
            for jp in 0..=w {
                let j = jp; // i0=1 => j = jp
                let dhi = dmx.min(jp as i32);
                for di in dlo..=dhi {
                    let d = di as usize;
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
            let (dmn_y, dmx_y) = band(y);
            let (dmn_z, dmx_z) = band(z);
            let dlo = dmn.max(0);
            for jp in 0..=w {
                let j = jp;
                let dhi = dmx.min(jp as i32);
                for di in dlo..=dhi {
                    let d = di as usize;
                    let idx = j * stride + d;
                    // In QDB, only split points k consistent with the bands of v,y,z
                    // (C inside_b, cm_dpsmall.c:4509-4531). k = right-fragment length.
                    let mut k = dmn_z.max(di - dmx_y);
                    if k < 0 { k = 0; }
                    let kmax = dmx_z.min(di - dmn_y);
                    if k > kmax {
                        a[idx] = IMPOSSIBLE;
                        ks[idx] = 0;
                        continue;
                    }
                    let ku = k as usize;
                    let mut best = alpha[y][(j - ku) * stride + (d - ku)] + alpha[z][j * stride + ku];
                    let mut bk = k;
                    for kk in (k + 1)..=kmax {
                        let kku = kk as usize;
                        let sc = alpha[y][(j - kku) * stride + (d - kku)] + alpha[z][j * stride + kku];
                        if sc > best {
                            best = sc;
                            bk = kk;
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
            let dlo = dmn.max(2); // d=0,1 impossible for MP (already IMPOSSIBLE)
            for jp in 0..=w {
                let j = jp;
                let dhi = dmx.min(jp as i32);
                for di in dlo..=dhi {
                    let d = di as usize;
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
            let dlo = dmn.max(1); // d=0 impossible
            for jp in 0..=w {
                let j = jp;
                let dhi = dmx.min(jp as i32);
                for di in dlo..=dhi {
                    let d = di as usize;
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
            let dlo = dmn.max(1); // d=0 impossible
            for jp in 0..=w {
                let j = jp;
                let dhi = dmx.min(jp as i32);
                for di in dlo..=dhi {
                    let d = di as usize;
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
    /// per-residue posterior-probability (`PP`) line (C ad->ppline). `None` when the
    /// hit was aligned without posteriors (`want_pp` FALSE); then the `Hit alignments`
    /// header shows the `cyksc` column instead of `acc`.
    pub ppline: Option<String>,
    /// display width == length of every annotation line (C ad->N). Used by the
    /// `cm_alidisplay_print` block-wrapping loop.
    pub n: usize,
    /// alignment score reported in the `cyksc` column when `ppline` is `None`
    /// (C ad->sc). Not shown in the default PP path.
    pub sc: f32,
    /// average posterior probability over aligned residues (C ad->avgpp); the `acc`
    /// column of the `Hit alignments` header line.
    pub avgpp: f32,
    /// hit GC fraction over the aligned residues (C ad->gc); `gc` column.
    pub gc: f32,
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
    /// cmsearch `-A` MSA support (approach B): the search parsetree that produced
    /// this display, retained so the `-A` writer can feed it straight through the
    /// byte-verified `parsetrees_to_alignment` (C reaches the same parsetree via
    /// cm_alidisplay_Backconvert; we reuse the original search parse directly).
    /// `None` for hits whose aligner path does not build a CM parsetree (hmm-only).
    pub ali_tr: Option<crate::parsetree::Parsetree>,
    /// Per-residue posterior-probability string for the retained parsetree (0-based,
    /// length L), or `None` when the hit was aligned without posteriors (e.g. `--qdb`
    /// CYK D&C). Fed as `ppstrs[i]` to `parsetrees_to_alignment`.
    pub ali_pp: Option<Vec<u8>>,
    /// The hit subsequence in 1-based sentinel-padded digital form (`[255, res.., 255]`),
    /// indexed by the retained parsetree's `emitl`/`emitr`. Fed as `sqdsq[i]`.
    pub ali_dsq: Option<Vec<u8>>,
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
    cyksc: f32,
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
    let model = String::from_utf8(model).unwrap();
    let n = model.len();

    // C cm_alidisplay.c:598-617 — ad->gc = (nC + nG)/span over the aligned residues
    // [emitl[0]..emitr[0]] (esl_abc_FCount; canonical residues here). ad->sc is the
    // parse (CYK) score adata->sc (cm_alidisplay.c:257); no PP ⇒ avgpp = 0.
    let mut cg = 0u32;
    let mut tot = 0u32;
    for x in tr.emitl[0]..=tr.emitr[0] {
        let c = dsq[x as usize];
        if c == 1 || c == 2 {
            cg += 1;
        }
        tot += 1;
    }
    let gc = cg as f32 / tot as f32;

    CmAliDisplay {
        aseq: String::from_utf8(aseq).unwrap(),
        csline: String::from_utf8(csline).unwrap(),
        ncline: String::from_utf8(ncline).unwrap(),
        model,
        mline: String::from_utf8(mline).unwrap(),
        rfline: String::from_utf8(rfline).unwrap(),
        ppline: None,
        n,
        sc: cyksc,
        avgpp: 0.0,
        gc,
        cfrom_emit,
        cto_emit,
        // J-mode / non-truncated: span == emit, no truncation, standard pass.
        cfrom_span: cfrom_emit,
        cto_span: cto_emit,
        trunc: "no".to_string(),
        pass_idx: 1,
        ali_tr: None,
        ali_pp: None,
        ali_dsq: None,
    }
}

/// Map the p7 HMM-only alidisplay ([`crate::p7_hmmonly::CmAliDisplayP7`], built by
/// `cm_alidisplay_CreateFromP7`) onto this crate's [`CmAliDisplay`] view for the
/// shared `Hit alignments` renderer + tblout. HMM-only hits carry no noncanonical
/// (NC) line (`ncline=""`), are J-mode (`cfrom_span==cfrom_emit`), always report a
/// PP line, `trunc="-"`, and `pass_idx=PLI_PASS_HMM_ONLY_ANY` (6).
pub fn cm_alidisplay_from_p7(p: &crate::p7_hmmonly::CmAliDisplayP7) -> CmAliDisplay {
    CmAliDisplay {
        aseq: p.aseq.clone(),
        csline: p.csline.clone(),
        ncline: String::new(),
        model: p.model.clone(),
        mline: p.mline.clone(),
        rfline: p.rfline.clone().unwrap_or_default(),
        ppline: p.ppline.clone(),
        n: p.n,
        sc: p.sc,
        avgpp: p.avgpp,
        gc: p.gc,
        cfrom_emit: p.cfrom_emit as i32,
        cto_emit: p.cto_emit as i32,
        cfrom_span: p.cfrom_span as i32,
        cto_span: p.cto_span as i32,
        trunc: "-".to_string(),
        pass_idx: crate::cm_trunc::PLI_PASS_HMM_ONLY_ANY,
        // HMM-only hits have no CM parsetree; `-A` cannot back them via approach (B).
        ali_tr: None,
        ali_pp: None,
        ali_dsq: None,
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

// ===========================================================================
// 4. cm_alidisplay_Create (FULL) — mode-aware, with PP + truncated local-end
//    markers. Port of cm_alidisplay.c:cm_alidisplay_Create (the default cmsearch
//    optacc+POST path). Used by the standard AND truncated pipeline passes.
// ===========================================================================

use crate::cp9::EmitMap;
use crate::cm_trunc::{TRMODE_J, TRMODE_L, TRMODE_R};

/// C `post_code_to_avg_pp` (cm_alidisplay.c:979).
#[inline]
fn post_code_to_avg_pp(pc: u8) -> f32 {
    match pc {
        b'*' => 0.975,
        b'9' => 0.9,
        b'8' => 0.8,
        b'7' => 0.7,
        b'6' => 0.6,
        b'5' => 0.5,
        b'4' => 0.4,
        b'3' => 0.3,
        b'2' => 0.2,
        b'1' => 0.1,
        b'0' => 0.025,
        _ => 0.0,
    }
}

/// C `cm_alidisplay_EncodePostProb` (cm_alidisplay.c:1099):
///   `return (p + 0.05 >= 1.0) ? '*' : (char) ((p + 0.05) * 10.) + '0';`
/// C's `p` is a `float`, but the literals `0.05`/`1.0`/`10.` are `double`, so the
/// `+ 0.05`, the `>= 1.0` compare, and the `* 10.` all happen in DOUBLE precision
/// (the float is promoted). Reproduce that exactly: doing the arithmetic in f32
/// mis-rounds the knife-edge (e.g. EL avg-PP 0.85 → C '8' via double vs f32 '9').
#[inline]
fn encode_post_prob(p: f32) -> u8 {
    let p = p as f64;
    if p + 0.05 >= 1.0 {
        b'*'
    } else {
        (((p + 0.05) * 10.0) as u8) + b'0'
    }
}

/// C `integer_textwidth` (cm_alidisplay.c:1074).
#[inline]
fn integer_textwidth(mut n: i64) -> i32 {
    let mut w = if n < 0 { 1 } else { 0 };
    while n != 0 {
        n /= 10;
        w += 1;
    }
    w
}

/// poor man's `(int)log_10(n)+1`, matching C's `do { w++; n/=10; } while(n)` — note
/// this returns 1 for n==0 (unlike integer_textwidth), exactly as C's EL/trunc width.
#[inline]
fn width_digits(mut n: i32) -> i32 {
    let mut w = 0;
    loop {
        w += 1;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    w
}

#[inline]
fn mode_emits_left(mode: i8) -> bool {
    mode == TRMODE_J || mode == TRMODE_L
}
#[inline]
fn mode_emits_right(mode: i8) -> bool {
    mode == TRMODE_J || mode == TRMODE_R
}

/// esl_abc_FAvgScore over a possibly-degenerate residue vs the singlet esc[] vector.
/// Canonical path: esc[di] (marginal averaging for degenerate codes, rare here).
#[inline]
fn favg_score(esc: &[f32], di: u8) -> f32 {
    singlet_score(esc, di)
}

enum PdaF {
    State(usize),
    /// a right-side residue to be emitted when popped (order matches C's push).
    Residue {
        rf: u8,
        nnc: u8,
        str_c: u8,
        cons: u8,
        mid: u8,
        seq: u8,
        post: u8,
    },
}

/// FULL port of `cm_alidisplay_Create` (cm_alidisplay.c:56). Builds all annotation
/// lines (rfline optional, ncline, csline, model, mline, aseq, ppline optional) with
/// mode-aware left/right emission, the PP line, and the 5'/3' truncated local-end
/// markers (`<[n]*` / `*[n]>`). `dsq` is the hit subsequence frame (`dsq[1..=lp]`).
/// `ppstr` (0-based `[0..lp-1]`) is the posterior code string from the aligner, or
/// `None` if aligned without posteriors. `sc`/`avgpp` come from the aligner.
#[allow(clippy::too_many_arguments)]
pub fn cm_alidisplay_create_full(
    cm: &CM,
    cmcons: &CmConsensus,
    emap: &EmitMap,
    tr: &Parsetree,
    ppstr: Option<&[u8]>,
    dsq: &[u8],
    lp: i32,
    pass_idx: i32,
    have_i0: bool,
    have_j0: bool,
    sc: f32,
    avgpp: f32,
) -> CmAliDisplay {
    let use_cons = !cm.consensus.is_empty(); // CMH_CONS
    let use_rf = !cm.rf.is_empty(); // CMH_RF
    let has_pp = ppstr.is_some();
    let m_state = cm.m; // EL sentinel state index

    // Truncation bounds (guessed span vs emit). Mirrors ParsetreeToCMBounds.
    let (cfrom_span, cto_span, cfrom_emit, cto_emit) =
        crate::cm_trunc::parsetree_to_cm_bounds(cm, emap, tr, pass_idx, have_i0, have_j0);

    // 5'/3' truncation marker widths (C:193-210).
    let (ntrunc_r, wtrunc_r) = if cfrom_span != cfrom_emit {
        let n = cfrom_emit - cfrom_span;
        (n, width_digits(n) + 4)
    } else {
        (0, 0)
    };
    let (ntrunc_l, wtrunc_l) = if cto_span != cto_emit {
        let n = cto_span - cto_emit;
        (n, width_digits(n) + 4)
    } else {
        (0, 0)
    };

    let mut rfline: Vec<u8> = Vec::new();
    let mut ncline: Vec<u8> = Vec::new();
    let mut csline: Vec<u8> = Vec::new();
    let mut model: Vec<u8> = Vec::new();
    let mut mline: Vec<u8> = Vec::new();
    let mut aseq: Vec<u8> = Vec::new();
    let mut ppline: Vec<u8> = Vec::new();

    // ---- 5' truncated begin marker "<[n]*" at the start (C:325-334) ----
    if ntrunc_r > 0 {
        let w = wtrunc_r as usize;
        let fw = (wtrunc_r - 4) as usize;
        for _ in 0..w {
            csline.push(b'~');
        }
        model.extend_from_slice(format!("<[{:>fw$}]*", ntrunc_r, fw = fw).as_bytes());
        aseq.extend_from_slice(format!("<[{:>fw$}]*", "0", fw = fw).as_bytes());
        for _ in 0..w {
            ncline.push(b' ');
            mline.push(b' ');
            if use_rf {
                rfline.push(b' ');
            }
            if has_pp {
                ppline.push(b'.');
            }
        }
    }

    // ---- main traceback (C:336-577) ----
    let mut pda: Vec<PdaF> = Vec::new();
    pda.push(PdaF::State(0));
    while let Some(item) = pda.pop() {
        match item {
            PdaF::Residue { rf, nnc, str_c, cons, mid, seq, post } => {
                if use_rf {
                    rfline.push(rf);
                }
                ncline.push(nnc);
                csline.push(str_c);
                model.push(cons);
                mline.push(mid);
                aseq.push(seq);
                if has_pp {
                    ppline.push(post);
                }
            }
            PdaF::State(ti) => {
                let v = tr.state[ti];
                // EL (local end, state M) — C:365-401.
                if v == m_state {
                    let prv_v = tr.state[ti - 1] as usize;
                    let nd = 1 + cm.ndidx[prv_v] as usize;
                    let qinset = cmcons.rpos[nd] - cmcons.lpos[nd] + 1;
                    let tinset = tr.emitr[ti] - tr.emitl[ti] + 1;
                    let ninset = qinset.max(tinset);
                    let numwidth = width_digits(ninset);
                    let nw = (numwidth + 4) as usize;
                    for _ in 0..nw {
                        csline.push(b'~');
                    }
                    model.extend_from_slice(
                        format!("*[{:>w$}]*", qinset, w = numwidth as usize).as_bytes(),
                    );
                    aseq.extend_from_slice(
                        format!("*[{:>w$}]*", tinset, w = numwidth as usize).as_bytes(),
                    );
                    for _ in 0..nw {
                        ncline.push(b' ');
                        mline.push(b' ');
                        if use_rf {
                            rfline.push(b'~');
                        }
                    }
                    if let Some(pp) = ppstr {
                        // average PP over EL emissions -> single code at slot numwidth+1.
                        let mut ppavg = 0.0f32;
                        for x in tr.emitl[ti]..=tr.emitr[ti] {
                            ppavg += post_code_to_avg_pp(pp[(x - 1) as usize]);
                        }
                        ppavg /= (tr.emitr[ti] - tr.emitl[ti] + 1) as f32;
                        let mut buf = vec![b'.'; nw];
                        if tr.emitl[ti] <= tr.emitr[ti] {
                            buf[numwidth as usize + 1] = encode_post_prob(ppavg);
                        }
                        ppline.extend_from_slice(&buf);
                    }
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
                let mode = tr.mode[ti];

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
                let mut lpost = b'.';
                let mut rpost = b'.';

                if stt == IL_ST {
                    if mode == TRMODE_J || mode == TRMODE_L {
                        do_left = true;
                        lrf = b'.';
                        lstr = b'.';
                        lcons = b'.';
                        lseq = sym(symi).to_ascii_lowercase();
                        if let Some(pp) = ppstr {
                            lpost = pp[(emitl - 1) as usize];
                        }
                    }
                } else if stt == IR_ST {
                    if mode == TRMODE_J || mode == TRMODE_R {
                        do_right = true;
                        rrf = b'.';
                        rstr = b'.';
                        rcons = b'.';
                        rseq = sym(symj).to_ascii_lowercase();
                        if let Some(pp) = ppstr {
                            rpost = pp[(emitr - 1) as usize];
                        }
                    }
                } else {
                    if (ndt == MATP_ND || ndt == MATL_ND)
                        && (mode == TRMODE_J || mode == TRMODE_L)
                    {
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
                            if let Some(pp) = ppstr {
                                lpost = pp[(emitl - 1) as usize];
                            }
                        } else {
                            lseq = b'-';
                            lpost = b'.';
                        }
                    }
                    if (ndt == MATP_ND || ndt == MATR_ND)
                        && (mode == TRMODE_J || mode == TRMODE_R)
                    {
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
                            if let Some(pp) = ppstr {
                                rpost = pp[(emitr - 1) as usize];
                            }
                        } else {
                            rseq = b'-';
                            rpost = b'.';
                        }
                    }
                }

                // mid + nnc lines (C:480-536)
                let mut lmid = b' ';
                let mut rmid = b' ';
                let mut lnnc = b' ';
                let mut rnnc = b' ';
                if stt == MP_ST {
                    if mode == TRMODE_L {
                        if lseq == lcons.to_ascii_uppercase() {
                            lmid = lseq;
                        }
                        lnnc = b'?';
                    } else if mode == TRMODE_R {
                        if rseq == rcons.to_ascii_uppercase() {
                            rmid = rseq;
                        }
                        rnnc = b'?';
                    } else if mode == TRMODE_J {
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
                    }
                } else if (stt == ML_ST || stt == IL_ST) && (mode == TRMODE_J || mode == TRMODE_L) {
                    if lseq == lcons.to_ascii_uppercase() {
                        lmid = lseq;
                    } else if favg_score(&cm.esc[vu], symi) > 0.0 {
                        lmid = b'+';
                    }
                } else if (stt == MR_ST || stt == IR_ST) && (mode == TRMODE_J || mode == TRMODE_R) {
                    if rseq == rcons.to_ascii_uppercase() {
                        rmid = rseq;
                    } else if favg_score(&cm.esc[vu], symj) > 0.0 {
                        rmid = b'+';
                    }
                }
                if (stid == MATP_ML || stid == MATP_MR) && mode == TRMODE_J {
                    lnnc = b'v';
                    rnnc = b'v';
                }
                if stid == MATP_ML && mode == TRMODE_L {
                    lnnc = b'?';
                }
                if stid == MATP_MR && mode == TRMODE_R {
                    rnnc = b'?';
                }

                if do_left {
                    if use_rf {
                        rfline.push(lrf);
                    }
                    ncline.push(lnnc);
                    csline.push(lstr);
                    model.push(lcons);
                    mline.push(lmid);
                    aseq.push(lseq);
                    if has_pp {
                        ppline.push(lpost);
                    }
                }
                if do_right {
                    pda.push(PdaF::Residue {
                        rf: rrf,
                        nnc: rnnc,
                        str_c: rstr,
                        cons: rcons,
                        mid: rmid,
                        seq: rseq,
                        post: rpost,
                    });
                }
                if tr.nxtr[ti] != -1 {
                    pda.push(PdaF::State(tr.nxtr[ti] as usize));
                }
                if tr.nxtl[ti] != -1 {
                    pda.push(PdaF::State(tr.nxtl[ti] as usize));
                }
            }
        }
    }

    // ---- 3' truncated begin marker "*[n]>" at the end (C:582-591) ----
    if ntrunc_l > 0 {
        let w = wtrunc_l as usize;
        let fw = (wtrunc_l - 4) as usize;
        for _ in 0..w {
            csline.push(b'~');
        }
        model.extend_from_slice(format!("*[{:>fw$}]>", ntrunc_l, fw = fw).as_bytes());
        aseq.extend_from_slice(format!("*[{:>fw$}]>", "0", fw = fw).as_bytes());
        for _ in 0..w {
            ncline.push(b' ');
            mline.push(b' ');
            if use_rf {
                rfline.push(b' ');
            }
            if has_pp {
                ppline.push(b'.');
            }
        }
    }

    // GC over the aligned residues (C:598-617).
    let mut cg = 0u32;
    let mut tot = 0u32;
    for x in tr.emitl[0]..=tr.emitr[0] {
        let c = dsq[x as usize];
        if c == 1 || c == 2 {
            cg += 1;
        }
        tot += 1;
    }
    let _ = lp;
    let gc = cg as f32 / tot as f32;

    let n = model.len();
    let trunc = trunc_string_owned(cfrom_span, cto_span, cfrom_emit, cto_emit);
    CmAliDisplay {
        aseq: String::from_utf8(aseq).unwrap(),
        csline: String::from_utf8(csline).unwrap(),
        ncline: String::from_utf8(ncline).unwrap(),
        model: String::from_utf8(model).unwrap(),
        mline: String::from_utf8(mline).unwrap(),
        rfline: String::from_utf8(rfline).unwrap(),
        ppline: if has_pp {
            Some(String::from_utf8(ppline).unwrap())
        } else {
            None
        },
        n,
        sc,
        avgpp,
        gc,
        cfrom_emit,
        cto_emit,
        cfrom_span,
        cto_span,
        trunc,
        pass_idx,
        // Retained by the caller (cm_search.rs sites) after construction when `-A`
        // support is needed; default None here.
        ali_tr: None,
        ali_pp: None,
        ali_dsq: None,
    }
}

fn trunc_string_owned(cfrom_span: i32, cto_span: i32, cfrom_emit: i32, cto_emit: i32) -> String {
    crate::cm_trunc::trunc_string(cfrom_span, cto_span, cfrom_emit, cto_emit).to_string()
}

/// FULL port of `cm_alidisplay_Print` (cm_alidisplay.c:1152). Wraps the alignment into
/// blocks of `aliwidth` columns (from `linewidth`/`min_aliwidth`), parsing the local-end
/// `*[n]*` / `<[n]*` / `*[n]>` runs so a block never splits one. `sqfrom`/`sqto` and the
/// name/acc strings are supplied by the caller (the pipeline's genomic coords + model).
#[allow(clippy::too_many_arguments)]
pub fn cm_alidisplay_print(
    ad: &CmAliDisplay,
    sqfrom: i64,
    sqto: i64,
    cmname: &str,
    cmacc: &str,
    sqname: &str,
    sqacc: &str,
    min_aliwidth: i32,
    linewidth: i32,
    show_accessions: bool,
) -> String {
    let show_cmname = if show_accessions && !cmacc.is_empty() { cmacc } else { cmname };
    let show_seqname = if show_accessions && !sqacc.is_empty() { sqacc } else { sqname };

    let namewidth = show_cmname.len().max(show_seqname.len()) as i32;
    let coordwidth = integer_textwidth(ad.cfrom_span as i64)
        .max(integer_textwidth(ad.cto_span as i64))
        .max(integer_textwidth(sqfrom))
        .max(integer_textwidth(sqto));
    let big_n = ad.n as i32;
    let mut aliwidth = if linewidth > 0 {
        linewidth - namewidth - 2 * coordwidth - 5
    } else {
        big_n
    };
    if aliwidth < big_n && aliwidth < min_aliwidth {
        aliwidth = min_aliwidth;
    }

    let ncb = ad.ncline.as_bytes();
    let csb = ad.csline.as_bytes();
    let mob = ad.model.as_bytes();
    let mib = ad.mline.as_bytes();
    let asb = ad.aseq.as_bytes();
    let ppb = ad.ppline.as_ref().map(|s| s.as_bytes());
    let rfb = if ad.rfline.is_empty() { None } else { Some(ad.rfline.as_bytes()) };

    let mut out = String::new();
    let mut i1 = sqfrom;
    let mut k1 = ad.cfrom_emit;

    let take = |src: &[u8], pos: usize, w: usize| -> String {
        let end = (pos + w).min(src.len());
        String::from_utf8_lossy(&src[pos..end]).into_owned()
    };

    let mut pos = 0i32;
    let mut cur_aliwidth = aliwidth;
    let mut first_block = true;
    while pos < big_n {
        if !first_block {
            out.push('\n');
        }
        first_block = false;
        let mut ni = 0i32;
        let mut nk = 0i32;
        cur_aliwidth = aliwidth;

        let mut z = pos;
        while z < pos + cur_aliwidth && z < big_n {
            let zu = z as usize;
            if (asb[zu] == b'*' && mob[zu] == b'*') || (asb[zu] == b'<' && mob[zu] == b'<') {
                let trunc_at_start = asb[zu] == b'<' && mob[zu] == b'<';
                let mut nk_toadd = 0i32;
                let mut ni_toadd = 0i32;
                // '[' must follow
                let mut zp = z + 2;
                while mob[zp as usize] == b' ' {
                    zp += 1;
                }
                while mob[zp as usize] != b']' {
                    nk_toadd = nk_toadd * 10 + (mob[zp as usize] - b'0') as i32;
                    zp += 1;
                }
                zp = z + 2;
                while asb[zp as usize] == b' ' {
                    zp += 1;
                }
                while asb[zp as usize] != b']' {
                    ni_toadd = ni_toadd * 10 + (asb[zp as usize] - b'0') as i32;
                    zp += 1;
                }
                if (zp + 1) >= (pos + aliwidth) {
                    cur_aliwidth = z - pos;
                } else {
                    nk += nk_toadd;
                    ni += ni_toadd;
                    if trunc_at_start {
                        k1 -= nk_toadd;
                    }
                    z = zp + 1;
                }
            } else {
                if mob[zu] != b'.' {
                    nk += 1;
                }
                if asb[zu] != b'-' {
                    ni += 1;
                }
            }
            z += 1;
        }

        let cw = cur_aliwidth as usize;
        let pad = (aliwidth - cur_aliwidth) as usize;
        let k2 = k1 + nk - 1;
        let (i2, i1_next);
        if sqfrom < sqto {
            i2 = i1 + ni as i64 - 1;
            i1_next = i1 + ni as i64;
        } else {
            i2 = i1 - ni as i64 + 1;
            i1_next = i1 - ni as i64;
        }
        let lblw = (namewidth + coordwidth + 1) as usize;
        let posu = pos as usize;

        // NC (C: printed only when ad->ncline != NULL; HMM-only hits have none)
        if !ncb.is_empty() {
            out.push_str(&format!(
                "  {:lblw$} {} {:pad$}NC\n",
                "",
                take(ncb, posu, cw),
                "",
                lblw = lblw,
                pad = pad
            ));
        }
        // CS
        out.push_str(&format!(
            "  {:lblw$} {} {:pad$}CS\n",
            "",
            take(csb, posu, cw),
            "",
            lblw = lblw,
            pad = pad
        ));
        // model line: "  %*s %*d %s %*s%-*d"  (name is RIGHT-justified in C)
        out.push_str(&format!(
            "  {:>nw$} {:>cwn$} {} {:pad$}{:<cwn$}\n",
            show_cmname,
            k1,
            take(mob, posu, cw),
            "",
            k2,
            nw = namewidth as usize,
            cwn = coordwidth as usize,
            pad = pad
        ));
        // mid line: "  %*s %s" with namewidth+coordwidth+1 blanks
        out.push_str(&format!(
            "  {:lblw$} {}\n",
            "",
            take(mib, posu, cw),
            lblw = lblw
        ));
        // aseq line
        if ni > 0 {
            out.push_str(&format!(
                "  {:>nw$} {:>cwn$} {} {:pad$}{:<cwn$}\n",
                show_seqname,
                i1,
                take(asb, posu, cw),
                "",
                i2,
                nw = namewidth as usize,
                cwn = coordwidth as usize,
                pad = pad
            ));
        } else {
            out.push_str(&format!(
                "  {:>nw$} {:>cwn$} {} {:pad$}{:>cwn$}\n",
                show_seqname,
                "-",
                take(asb, posu, cw),
                "",
                "-",
                nw = namewidth as usize,
                cwn = coordwidth as usize,
                pad = pad
            ));
        }
        // PP
        if let Some(pp) = ppb {
            out.push_str(&format!(
                "  {:lblw$} {} {:pad$}PP\n",
                "",
                take(pp, posu, cw),
                "",
                lblw = lblw,
                pad = pad
            ));
        }
        // RF
        if let Some(rf) = rfb {
            out.push_str(&format!(
                "  {:lblw$} {} {:pad$}RF\n",
                "",
                take(rf, posu, cw),
                "",
                lblw = lblw,
                pad = pad
            ));
        }

        k1 += nk;
        i1 = i1_next;
        pos += cur_aliwidth;
    }
    out
}
