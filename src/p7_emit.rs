// SPDX-License-Identifier: BSD-3-Clause
//! Faithful port of HMMER's `p7_emit.c` sampling routines, as driven by
//! Infernal 1.1.5 `cmemit --hmmonly`:
//!
//!   * `p7_ProfileEmit`        (emit.c) — sample an unaligned sequence from a
//!                                        configured profile (used by `-u`).
//!   * `p7_CoreEmit`           (emit.c) — sample a sequence + trace from the core
//!                                        HMM, glocal-style (used by `-a`).
//!   * `p7_emit_FancyConsensus`(emit.c) — the HMM consensus sequence (used by `-c`).
//!
//! RNG parity is the load-bearing invariant: every `esl_rnd_FChoose` /
//! `esl_rnd_Roll` draw is reproduced in the exact same order as C, using the
//! shared `EslRandomFast` (esl_randomness_CreateFast) stream, so a fixed `--seed`
//! reproduces C's emitted sequences byte-for-byte.
//!
//! cmemit configures the filter HMM with `p7_ProfileConfig(hmm, bg, gm,
//! fp7->max_length, hmm_mode)` where `hmm_mode = -l ? p7_UNILOCAL : p7_UNIGLOCAL`
//! (cmemit.c:339), then forces the flanking length model OFF:
//! ```c
//! gm->xsc[p7P_N][p7P_MOVE] = gm->xsc[p7P_C][p7P_MOVE] = 0.0f;         // prob 1
//! gm->xsc[p7P_N][p7P_LOOP] = gm->xsc[p7P_C][p7P_LOOP] = -eslINFINITY; // prob 0
//! ```
//! Because the mode is *uni*hit, E→C=1 / E→J=0, and J is never reached. So every
//! special-state distribution collapses to `[LOOP=0, MOVE=1]`: each of N, E, C is
//! visited exactly once and each `FChoose` there deterministically returns MOVE —
//! but still consumes one RNG draw, which we must reproduce.

use crate::cm_emit::{esl_rnd_fchoose, EslRandomFast};
use crate::p7_generic::build_local_profile;
use crate::p7_hmm::P7Profile;
use crate::easel::msa::EslMsa;

// p7 trace state types (hmmer.h enum p7t_statetype_e).
const P7T_M: u8 = 1;
const P7T_D: u8 = 2;
const P7T_I: u8 = 3;
const P7T_S: u8 = 4;
const P7T_N: u8 = 5;
const P7T_B: u8 = 6;
const P7T_E: u8 = 7;
const P7T_C: u8 = 8;
const P7T_T: u8 = 9;
const P7T_J: u8 = 10;

const K_ABC: usize = 4; // RNA alphabet size (cm->abc->K)

/// A sampled p7 trace, mirroring the fields of C's `P7_TRACE` that
/// `p7_tracealign_Seqs` reads: parallel `st[]`, `k[]`, `i[]` arrays plus `M`/`L`.
#[derive(Debug, Clone)]
pub struct P7Trace {
    pub st: Vec<u8>,
    pub k: Vec<i32>,
    pub i: Vec<i32>,
    pub m: i32,
    pub l: i32,
}

impl P7Trace {
    fn new() -> Self {
        P7Trace { st: Vec::new(), k: Vec::new(), i: Vec::new(), m: 0, l: 0 }
    }
    #[inline]
    fn append(&mut self, st: u8, k: i32, i: i32) {
        self.st.push(st);
        self.k.push(k);
        self.i.push(i);
    }
}

/// C: emit.c `p7_ProfileEmit(r, hmm, gm, bg, sq, tr)` — sample an unaligned
/// sequence from a configured profile. Returns the emitted residues (canonical
/// indices `0..K-1`); the trace is not needed by cmemit's `-u` output so it is
/// not materialised (its construction consumes no RNG).
///
/// `is_local` selects UNILOCAL (`true`, from `-l`) vs UNIGLOCAL (`false`). In
/// local mode the B state enters via `sample_endpoints` (occupancy-weighted
/// B→Mk distribution from a LOCAL profile); in glocal mode B uses the core
/// `hmm->t[0]` transitions directly.
pub fn p7_profile_emit(r: &mut EslRandomFast, hmm: &P7Profile, is_local: bool) -> Vec<u8> {
    let m = hmm.m as usize;

    // Precompute the local B→Mk entry distribution `pstart` if needed
    // (sample_endpoints, emit.c). For glocal we never touch it.
    // pstart[0] = 0; pstart[k] = exp(TSC(k-1,BM)) * (M-k+1), k=1..M.
    let pstart: Vec<f32> = if is_local {
        let gm = build_local_profile(hmm, hmm.max_length);
        let mut ps = vec![0.0f32; m + 1];
        for k in 1..=m {
            let bm = gm.tsc[(k - 1) * 8 + 3]; // P7P_NTRANS=8, P7P_BM=3
            ps[k] = ((bm as f64).exp() * ((m - k + 1) as f64)) as f32;
        }
        ps
    } else {
        Vec::new()
    };

    // Special-state distributions collapse to [LOOP=0, MOVE=1] (see module docs):
    // deterministic MOVE, one RNG draw each.
    let xt_move: [f32; 2] = [0.0, 1.0];
    // Background residue distribution for N/C/J self-loops (never taken here, but
    // faithful): p7_bg RNA f[x] = 0.25.
    let bgf: [f32; 4] = [0.25; 4];

    let mut seq: Vec<u8> = Vec::new();
    let mut k: usize = 0;
    let mut kend: usize = m;
    let mut st: u8 = P7T_N; // C: st = p7T_N; i = 0; while (st != p7T_T)
    loop {
        let prv = st;
        match st {
            P7T_B => {
                if is_local {
                    // sample_endpoints: kstart ~ pstart; kend uniform in exits.
                    let kstart = esl_rnd_fchoose(r, &pstart, m + 1);
                    let roll = r.roll((m - kstart + 1) as u32) as usize;
                    k = kstart;
                    kend = kstart + roll;
                    st = P7T_M; // left wing retracted
                } else {
                    // glocal: B as M_0, use its MID transitions.
                    match esl_rnd_fchoose(r, &hmm.trans[0][0..3], 3) {
                        0 => { st = P7T_M; k = 1; }
                        1 => { st = P7T_I; k = 0; }
                        _ => { st = P7T_D; k = 1; }
                    }
                }
            }
            P7T_M => {
                if k == kend {
                    st = P7T_E; // preordained fate
                } else {
                    match esl_rnd_fchoose(r, &hmm.trans[k][0..3], 3) {
                        0 => st = P7T_M,
                        1 => st = P7T_I,
                        _ => st = P7T_D,
                    }
                }
            }
            P7T_D => {
                if k == kend {
                    st = P7T_E;
                } else {
                    st = if esl_rnd_fchoose(r, &hmm.trans[k][5..7], 2) == 0 { P7T_M } else { P7T_D };
                }
            }
            P7T_I => {
                st = if esl_rnd_fchoose(r, &hmm.trans[k][3..5], 2) == 0 { P7T_M } else { P7T_I };
            }
            P7T_N => {
                st = if esl_rnd_fchoose(r, &xt_move, 2) == 1 { P7T_B } else { P7T_N };
            }
            P7T_E => {
                st = if esl_rnd_fchoose(r, &xt_move, 2) == 1 { P7T_C } else { P7T_J };
            }
            P7T_C => {
                st = if esl_rnd_fchoose(r, &xt_move, 2) == 1 { P7T_T } else { P7T_C };
            }
            P7T_J => {
                st = if esl_rnd_fchoose(r, &xt_move, 2) == 1 { P7T_B } else { P7T_J };
            }
            _ => unreachable!("impossible state in p7_profile_emit"),
        }

        // Update k based on the transition just sampled.
        if st == P7T_E {
            k = 0;
        } else if st == P7T_M && prv != P7T_B {
            k += 1;
        } else if st == P7T_D {
            k += 1;
        }

        // Generate a residue based on the transition just sampled.
        if st == P7T_M {
            seq.push(esl_rnd_fchoose(r, &hmm.mat[k][0..K_ABC], K_ABC) as u8);
        } else if st == P7T_I {
            seq.push(esl_rnd_fchoose(r, &hmm.ins[k][0..K_ABC], K_ABC) as u8);
        } else if (st == P7T_N || st == P7T_C || st == P7T_J) && prv == st {
            seq.push(esl_rnd_fchoose(r, &bgf, K_ABC) as u8);
        }

        if st == P7T_T {
            break;
        }
    }
    seq
}

/// C: emit.c `p7_CoreEmit(r, hmm, sq, tr)` — sample a sequence + trace from the
/// core probability model (glocal, no special states). Used by cmemit `-a`
/// (aligned output), which always emits with `p7_CoreEmit` regardless of `-l`.
/// Returns `(trace, residues)` where `residues` are canonical indices `0..K-1`.
pub fn p7_core_emit(r: &mut EslRandomFast, hmm: &P7Profile) -> (P7Trace, Vec<u8>) {
    let m = hmm.m as usize;
    let mut tr = P7Trace::new();
    let mut seq: Vec<u8> = Vec::new();

    let mut k: usize = 0;
    let mut i: i32 = 0;
    let mut st: u8 = P7T_B;
    tr.append(st, k as i32, i); // C appends the initial B state

    while st != P7T_E {
        // Sample next state type given current state (and current k).
        match st {
            P7T_B | P7T_M => {
                match esl_rnd_fchoose(r, &hmm.trans[k][0..3], 3) {
                    0 => st = P7T_M,
                    1 => st = P7T_I,
                    _ => st = P7T_D,
                }
            }
            P7T_I => {
                st = if esl_rnd_fchoose(r, &hmm.trans[k][3..5], 2) == 0 { P7T_M } else { P7T_I };
            }
            P7T_D => {
                st = if esl_rnd_fchoose(r, &hmm.trans[k][5..7], 2) == 0 { P7T_M } else { P7T_D };
            }
            _ => unreachable!("impossible state in p7_core_emit"),
        }

        // Bump k,i depending on new state type.
        if st == P7T_M || st == P7T_D {
            k += 1;
        }
        if st == P7T_M || st == P7T_I {
            i += 1;
        }

        // A transit to M_{M+1} is a transit to the E state.
        if k == m + 1 {
            if st == P7T_M {
                st = P7T_E;
                k = 0;
            } else {
                unreachable!("failed to reach E state properly");
            }
        }

        // Sample residue if in match or insert.
        let x: Option<usize> = if st == P7T_M {
            Some(esl_rnd_fchoose(r, &hmm.mat[k][0..K_ABC], K_ABC))
        } else if st == P7T_I {
            Some(esl_rnd_fchoose(r, &hmm.ins[k][0..K_ABC], K_ABC))
        } else {
            None
        };

        tr.append(st, k as i32, i);
        if let Some(xx) = x {
            seq.push(xx as u8);
        }
    }

    tr.m = m as i32;
    tr.l = i;
    (tr, seq)
}

/// C: emit.c `p7_emit_FancyConsensus(hmm, min_lower, min_upper, sq)`. cmemit
/// calls it with `min_lower=0.0, min_upper=0.5` (cmemit.c:627). Returns the text
/// consensus (one char per match node 1..M). No RNG. `hmm->mm` is never set for
/// infernox filter HMMs, so the masked-position branch is unreachable.
pub fn p7_emit_fancy_consensus(hmm: &P7Profile, min_lower: f32, min_upper: f32) -> Vec<u8> {
    const SYM: [u8; 4] = [b'A', b'C', b'G', b'U'];
    let m = hmm.m as usize;
    let mut out = Vec::with_capacity(m);
    for k in 1..=m {
        // p = esl_vec_FMax(mat[k]); x = esl_vec_FArgMax(mat[k]).
        let mut x = 0usize;
        let mut p = hmm.mat[k][0];
        for j in 1..K_ABC {
            if hmm.mat[k][j] > p {
                p = hmm.mat[k][j];
                x = j;
            }
        }
        let c = if p < min_lower {
            b'n' // tolower(unknown) — unreachable with min_lower=0.0
        } else if p >= min_upper {
            SYM[x] // toupper
        } else {
            SYM[x].to_ascii_lowercase()
        };
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------
// p7_tracealign_Seqs and helpers (HMMER tracealign.c), for cmemit --hmmonly -a.
// cmemit calls p7_tracealign_Seqs(sqA, p7trA, nseq, fp7->M, p7_ALL_CONSENSUS_COLS,
// fp7, ret_msa): optflags = p7_ALL_CONSENSUS_COLS only (no p7_TRIM, no
// p7_DIGITIZE), so all match columns are used and the output MSA is text mode.
// The sqA seqs are created with fp7->abc (RNA) and never re-aliased to the output
// alphabet, so HMM -a output is always RNA (unlike the CM -a path, --dna does not
// apply here). The p7 traces carry no posterior probabilities and fp7 has no mm
// mask, so annotate_posterior_probability / annotate_mm are no-ops (omitted).

const SYM4: [u8; 4] = [b'A', b'C', b'G', b'U'];

#[inline]
fn is_text_gap(c: u8) -> bool {
    // esl_abc_CIsGap: '-', '.', '_' (and '~' missing-data) count as gaps.
    c == b'-' || c == b'.' || c == b'_' || c == b'~'
}
#[inline]
fn is_text_residue(c: u8) -> bool {
    // esl_abc_CIsResidue: an alphabetic symbol that is not a gap/missing char.
    c.is_ascii_alphabetic()
}

/// C: tracealign.c `map_new_msa` with optflags = p7_ALL_CONSENSUS_COLS. Returns
/// `(inscount[0..=M], matuse[0..=M], matmap[0..=M], alen)`.
fn map_new_msa(trs: &[P7Trace], m: usize) -> (Vec<i32>, Vec<bool>, Vec<usize>, usize) {
    let mut inscount = vec![0i32; m + 1];
    let mut matuse = vec![true; m + 1];
    matuse[0] = false; // matuse[0]=0; matuse[1..M]=TRUE (ALL_CONSENSUS_COLS)
    let mut matmap = vec![0usize; m + 1];

    for tr in trs {
        let mut insnum = vec![0i32; m + 1];
        for z in 1..tr.st.len() {
            match tr.st[z] {
                P7T_I => insnum[tr.k[z] as usize] += 1,
                P7T_N => {
                    if tr.st[z - 1] == P7T_N {
                        insnum[0] += 1;
                    }
                }
                P7T_C => {
                    if tr.st[z - 1] == P7T_C {
                        insnum[m] += 1;
                    }
                }
                P7T_M => matuse[tr.k[z] as usize] = true,
                _ => {}
            }
        }
        for k in 0..=m {
            inscount[k] = inscount[k].max(insnum[k]);
        }
    }

    // (no p7_TRIM) set matmap[] from inscount/matuse.
    let mut alen = inscount[0] as usize;
    for k in 1..=m {
        if matuse[k] {
            matmap[k] = alen + 1;
            alen += 1 + inscount[k] as usize;
        } else {
            matmap[k] = alen;
            alen += inscount[k] as usize;
        }
    }
    (inscount, matuse, matmap, alen)
}

/// C: tracealign.c `make_text_msa` (text mode, no TRIM). Builds each row of the
/// alignment as bytes. `seqs[idx]` holds canonical residue indices (0-based) for
/// sequence positions 1..n; `get_dsq_z(idx,z) = seqs[idx][tr.i[z]-1]`.
fn make_text_rows(
    trs: &[P7Trace],
    seqs: &[Vec<u8>],
    matuse: &[bool],
    matmap: &[usize],
    m: usize,
    alen: usize,
) -> Vec<Vec<u8>> {
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(trs.len());
    for (idx, tr) in trs.iter().enumerate() {
        let mut row = vec![b'.'; alen];
        for k in 1..=m {
            if matuse[k] {
                row[matmap[k] - 1] = b'-';
            }
        }
        let mut apos: usize = 0;
        for z in 0..tr.st.len() {
            let k = tr.k[z] as usize;
            let i = tr.i[z];
            match tr.st[z] {
                P7T_M => {
                    let x = seqs[idx][(i - 1) as usize] as usize;
                    row[matmap[k] - 1] = SYM4[x].to_ascii_uppercase();
                    apos = matmap[k];
                }
                P7T_D => {
                    if matuse[k] {
                        row[matmap[k] - 1] = b'-';
                    }
                    apos = matmap[k];
                }
                P7T_I => {
                    let x = seqs[idx][(i - 1) as usize] as usize;
                    row[apos] = SYM4[x].to_ascii_lowercase();
                    apos += 1;
                }
                P7T_N | P7T_C => {
                    if i > 0 {
                        let x = seqs[idx][(i - 1) as usize] as usize;
                        row[apos] = SYM4[x].to_ascii_lowercase();
                        apos += 1;
                    }
                }
                P7T_E => {
                    apos = matmap[m];
                }
                _ => {}
            }
        }
        rows.push(row);
    }
    rows
}

/// C: tracealign.c `annotate_rf` — '.' everywhere, 'x' at each used match column.
fn annotate_rf(matuse: &[bool], matmap: &[usize], m: usize, alen: usize) -> Vec<u8> {
    let mut rf = vec![b'.'; alen];
    for k in 1..=m {
        if matuse[k] {
            rf[matmap[k] - 1] = b'x';
        }
    }
    rf
}

/// C: tracealign.c `rejustify_insertions_text`. Splits each insert block in half
/// (left- and right-justified), operating on the text rows in place. (No pp.)
fn rejustify_insertions_text(
    rows: &mut [Vec<u8>],
    inserts: &[i32],
    matmap: &[usize],
    matuse: &[bool],
    m: usize,
) {
    for row in rows.iter_mut() {
        for k in 0..m {
            if inserts[k] > 1 {
                let mu_k1 = if matuse[k + 1] { 1usize } else { 0 };
                let region_end = matmap[k + 1] - mu_k1; // exclusive upper bound (0-based)
                let mut nins = 0usize;
                let mut apos = matmap[k];
                while apos < region_end {
                    if is_text_residue(row[apos]) {
                        nins += 1;
                    }
                    apos += 1;
                }
                if k == 0 {
                    nins = 0; // N-terminus is right justified
                } else {
                    nins /= 2; // split in half
                }
                let lo = matmap[k] + nins;
                // opos, npos are signed to allow the >= lo comparison to fail.
                let mut opos: isize = (region_end as isize) - 1;
                let mut npos: isize = (region_end as isize) - 1;
                while opos >= lo as isize {
                    if is_text_gap(row[opos as usize]) {
                        opos -= 1;
                    } else {
                        row[npos as usize] = row[opos as usize];
                        npos -= 1;
                        opos -= 1;
                    }
                }
                while npos >= lo as isize {
                    row[npos as usize] = b'.';
                    npos -= 1;
                }
            }
        }
    }
}

/// C: tracealign.c `p7_tracealign_Seqs(sq, tr, nseq, M, p7_ALL_CONSENSUS_COLS,
/// hmm, ret_msa)` for core traces from p7_CoreEmit. Returns a text-mode EslMsa
/// with per-seq names, `#=GC RF`, and unit weights. cmemit then adds
/// `#=GC SS_cons`, the alignment name (#=GF ID) and description (#=GF DE).
pub fn p7_tracealign_seqs(
    trs: &[P7Trace],
    seqs: &[Vec<u8>],
    names: &[String],
    m: usize,
) -> EslMsa {
    let (inscount, matuse, matmap, alen) = map_new_msa(trs, m);
    let mut rows = make_text_rows(trs, seqs, &matuse, &matmap, m, alen);
    let rf = annotate_rf(&matuse, &matmap, m, alen);
    rejustify_insertions_text(&mut rows, &inscount, &matmap, &matuse, m);

    let nseq = trs.len();
    let mut msa = EslMsa::new();
    msa.is_digital = false;
    msa.alen = alen as i64;
    msa.nseq = nseq;
    for idx in 0..nseq {
        msa.aseq.push(String::from_utf8(rows[idx].clone()).unwrap());
        msa.sqname.push(names[idx].clone());
        msa.wgt.push(1.0);
    }
    msa.rf = Some(String::from_utf8(rf).unwrap());
    msa
}
