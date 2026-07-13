//! Parse tree structure for CM alignment traceback
//!
//! A parse tree represents the alignment of a sequence to a covariance model,
//! showing which states were visited and which residues were emitted.

/// Parse tree node representing one step in the alignment
///
/// The parse tree is a doubly-linked tree structure where each node
/// represents a state visit in the CM alignment. Nodes are connected
/// via nxtl (next left sibling), nxtr (next right sibling), and prv (parent).
#[derive(Debug, Clone)]
pub struct Parsetree {
    /// Number of nodes in the tree
    pub n: i32,
    /// Left sequence position emitted (1-indexed, 0 if no emission)
    pub emitl: Vec<i32>,
    /// Right sequence position emitted (1-indexed, 0 if no emission)
    pub emitr: Vec<i32>,
    /// State index in the CM
    pub state: Vec<i32>,
    /// Next left child index (-1 if none)
    pub nxtl: Vec<i32>,
    /// Next right child index (-1 if none)
    pub nxtr: Vec<i32>,
    /// Previous node index (-1 if root)
    pub prv: Vec<i32>,
    /// Per-node truncation marginal mode (C `tr->mode`): TRMODE_J(3), TRMODE_L(2),
    /// TRMODE_R(1), TRMODE_T(0). Defaults to TRMODE_J for non-truncated parses.
    pub mode: Vec<i8>,
    /// C `tr->is_std`: TRUE for a standard (non-truncated) parse; the truncated
    /// aligners lower this. Used by Parsetrees2Alignment allow_trunc (cmbuild --miss).
    pub is_std: bool,
    /// C `tr->pass_idx`: pipeline pass index that produced this parse; only
    /// meaningful when `is_std == false` (drives PassEnforcesFirst/FinalRes).
    pub pass_idx: i32,
    /// C `tr->trpenalty`: the truncated-begin penalty assessed for this parse
    /// (`(local? l_ptyAA : g_ptyAA)[pty_idx][b]`, cm_dpalign_trunc.c:233). 0.0 for
    /// standard parses; used by ParsetreeScore (adds it when !is_std) and the
    /// --tfile ParsetreeDump header.
    pub trpenalty: f32,
}

impl Parsetree {
    /// Create new parsetree with given capacity
    pub fn new(capacity: usize) -> Self {
        Parsetree {
            n: 0,
            emitl: Vec::with_capacity(capacity),
            emitr: Vec::with_capacity(capacity),
            state: Vec::with_capacity(capacity),
            nxtl: Vec::with_capacity(capacity),
            nxtr: Vec::with_capacity(capacity),
            prv: Vec::with_capacity(capacity),
            mode: Vec::with_capacity(capacity),
            // C CreateParsetree (cm_parsetree.c:68-70): is_std = TRUE, pass_idx =
            // PLI_PASS_STD_ANY (1), trpenalty = 0. The truncated aligners lower is_std
            // and set pass_idx/trpenalty; standard parses keep these defaults.
            is_std: true,
            pass_idx: 1, // PLI_PASS_STD_ANY
            trpenalty: 0.0,
        }
    }

    /// Add a node to the parsetree
    ///
    /// # Arguments
    /// * `emitl` - Left sequence position (1-indexed, 0 if no left emission)
    /// * `emitr` - Right sequence position (1-indexed, 0 if no right emission)
    /// * `state` - State index in the CM
    /// * `nxtl` - Index of next left child (-1 if none)
    /// * `nxtr` - Index of next right child (-1 if none)
    /// * `prv` - Index of parent node (-1 if root)
    ///
    /// # Returns
    /// The index of the newly added node
    pub fn add_node(
        &mut self,
        emitl: i32,
        emitr: i32,
        state: i32,
        nxtl: i32,
        nxtr: i32,
        prv: i32,
    ) -> i32 {
        // Default marginal mode is TRMODE_J (3) — correct for non-truncated parses.
        self.add_node_mode(emitl, emitr, state, nxtl, nxtr, prv, 3)
    }

    /// Add a node carrying an explicit truncation marginal mode (C
    /// `InsertTraceNodewithMode`). See [`Parsetree::add_node`] for the shared args.
    #[allow(clippy::too_many_arguments)]
    pub fn add_node_mode(
        &mut self,
        emitl: i32,
        emitr: i32,
        state: i32,
        nxtl: i32,
        nxtr: i32,
        prv: i32,
        mode: i8,
    ) -> i32 {
        let idx = self.n;
        self.emitl.push(emitl);
        self.emitr.push(emitr);
        self.state.push(state);
        self.nxtl.push(nxtl);
        self.nxtr.push(nxtr);
        self.prv.push(prv);
        self.mode.push(mode);
        self.n += 1;
        idx
    }

    /// Get the root node index (always 0)
    pub fn root(&self) -> Option<i32> {
        if self.n > 0 {
            Some(0)
        } else {
            None
        }
    }

    /// Get state at node index
    pub fn get_state(&self, idx: usize) -> Option<i32> {
        if idx < self.state.len() {
            Some(self.state[idx])
        } else {
            None
        }
    }

    /// Get emissions at node index (emitl, emitr)
    pub fn get_emissions(&self, idx: usize) -> Option<(i32, i32)> {
        if idx < self.emitl.len() {
            Some((self.emitl[idx], self.emitr[idx]))
        } else {
            None
        }
    }

    /// Get hit boundaries from the parsetree
    ///
    /// Returns (start, end) positions in the sequence (1-indexed).
    /// Scans all nodes to find the minimum left emission and maximum right emission.
    pub fn get_hit_bounds(&self) -> Option<(i32, i32)> {
        if self.n == 0 {
            return None;
        }

        let mut min_l = i32::MAX;
        let mut max_r = i32::MIN;

        for i in 0..self.n as usize {
            let l = self.emitl[i];
            let r = self.emitr[i];
            // Only consider actual emissions (non-zero positions)
            if l > 0 && l < min_l {
                min_l = l;
            }
            if r > 0 && r > max_r {
                max_r = r;
            }
        }

        if min_l == i32::MAX || max_r == i32::MIN {
            // No emissions found (shouldn't happen in valid parsetree)
            Some((1, 1))
        } else {
            Some((min_l, max_r))
        }
    }
}

// ---------------------------------------------------------------------------
// Parsetree scoring + dump (C cm_parsetree.c). Truncation-aware, used by cmalign
// --tfile. Shared here rather than binary-local so both the SCORE line and the
// per-node table match C exactly for standard and truncated parses.
// ---------------------------------------------------------------------------

use crate::cm::{abc_favg_score, abc_fcount_frac, ALPHABET_SIZE, ALPHABET_SIZE_P, CM};
use crate::cm_trunc::{TRMODE_J, TRMODE_L, TRMODE_R};
use crate::constants::{B_ST, E_ST, EL_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST};

#[inline]
fn mode_emits_left(mode: i8) -> bool {
    mode == TRMODE_J || mode == TRMODE_L
}
#[inline]
fn mode_emits_right(mode: i8) -> bool {
    mode == TRMODE_J || mode == TRMODE_R
}

/// Per-sequence insert / EL-insert info for cmalign --ifile / --elfile. C computes
/// this inside Parsetrees2Alignment (cm_parsetree.c:1085-1219) while assembling the
/// MSA; we reproduce just the accounting walk (no aseq build) so it can be emitted
/// without perturbing the byte-verified alignment builder. Returns the two per-seq
/// lines (the `<seqname> <seqlen> <spos> <epos> [<c> <u> <i>]...` records for the
/// insert file and the EL file; a line has only the 4-token prefix if no inserts).
pub fn insert_el_info_lines(
    cm: &CM,
    emap: &crate::cp9::EmitMap,
    tr: &Parsetree,
    name: &str,
    seqlen: i64,
) -> (String, String) {
    let clen = emap.clen as usize;
    let m = cm.m;
    let mut iluse = vec![0i32; clen + 1];
    let mut iruse = vec![0i32; clen + 1];
    let mut eluse = vec![0i32; clen + 1];
    let mut ifirst = vec![-1i32; clen + 1];
    let mut elfirst = vec![-1i32; clen + 1];
    let mut s_cpos = emap.clen + 1;
    let mut e_cpos = 0i32;
    let mut prvnd = 0i32;

    for tpos in 0..tr.n as usize {
        let v = tr.state[tpos];
        let mode = tr.mode[tpos];
        let stt = if v == m { EL_ST } else { cm.sttype[v as usize] as i32 };
        // C: EL uses the previous node (nd = prvnd); else nd = ndidx[v].
        let nd = if stt == EL_ST { prvnd } else { cm.ndidx[v as usize] } as usize;

        if stt == MP_ST {
            if mode_emits_left(mode) {
                let cpos = emap.lpos[nd];
                s_cpos = s_cpos.min(cpos);
                e_cpos = e_cpos.max(cpos);
            }
            if mode_emits_right(mode) {
                let cpos = emap.rpos[nd];
                s_cpos = s_cpos.min(cpos);
                e_cpos = e_cpos.max(cpos);
            }
        } else if stt == ML_ST {
            if mode_emits_left(mode) {
                let cpos = emap.lpos[nd];
                s_cpos = s_cpos.min(cpos);
                e_cpos = e_cpos.max(cpos);
            }
        } else if stt == MR_ST {
            if mode_emits_right(mode) {
                let cpos = emap.rpos[nd];
                s_cpos = s_cpos.min(cpos);
                e_cpos = e_cpos.max(cpos);
            }
        } else if stt == IL_ST {
            if mode_emits_left(mode) {
                let cpos = emap.lpos[nd] as usize;
                let rpos = tr.emitl[tpos];
                if iluse[cpos] == 0 {
                    ifirst[cpos] = rpos; // first insert for this IL
                }
                iluse[cpos] += 1;
            }
        } else if stt == EL_ST {
            let cpos = emap.epos[nd] as usize;
            eluse[cpos] = tr.emitr[tpos] - tr.emitl[tpos] + 1;
            elfirst[cpos] = tr.emitl[tpos];
        } else if stt == IR_ST && mode_emits_right(mode) {
            let cpos = (emap.rpos[nd] - 1) as usize; // -1 => "following this one"
            let rpos = tr.emitr[tpos];
            ifirst[cpos] = rpos; // overwrite -> ends up min rpos (IR written 3'->5')
            iruse[cpos] += 1;
        }
        prvnd = nd as i32;
    }

    let s_cposa = if s_cpos == emap.clen + 1 { -1 } else { s_cpos };
    let e_cposa = if e_cpos == 0 { -1 } else { e_cpos };

    let mut iline = format!("{} {} {} {}", name, seqlen, s_cposa, e_cposa);
    let mut eline = format!("{} {} {} {}", name, seqlen, s_cposa, e_cposa);
    for cpos in 0..=clen {
        if iluse[cpos] + iruse[cpos] > 0 {
            iline.push_str(&format!("  {} {} {}", cpos, ifirst[cpos], iluse[cpos] + iruse[cpos]));
        }
        if eluse[cpos] > 0 {
            eline.push_str(&format!("  {} {} {}", cpos, elfirst[cpos], eluse[cpos]));
        }
    }
    (iline, eline)
}

/// C eslRNA alphabet symbols (esl_alphabet.c set_type), codes 0..17.
const RNA_SYM: &[u8; ALPHABET_SIZE_P] = b"ACGU-RYMKSWHBVDN*~";

/// C `CMH_LOCAL_BEGIN` — the runtime "local begin is on" flag set by cm_Configure
/// (cm_alndata.rs sets `1<<10`; cm_dpalign uses the same). NOT `cm::CM_LOCAL_BEGIN`
/// (1<<0), which is a mislabeled constant this codebase's DP path does not use.
const CMH_LOCAL_BEGIN: u32 = 1 << 10;

/// C StateDelta(sttype) (cm.c:1180): #residues emitted (MP=2, ML/MR/IL/IR=1, else 0).
#[inline]
fn state_delta_of(stt: i32) -> i32 {
    if stt == MP_ST {
        2
    } else if stt == ML_ST || stt == MR_ST || stt == IL_ST || stt == IR_ST {
        1
    } else {
        0
    }
}
/// C StateLeftDelta(sttype) (cm.c): 1 if the state emits left (MP/ML/IL), else 0.
#[inline]
fn sdl_of(stt: i32) -> i32 {
    if stt == MP_ST || stt == ML_ST || stt == IL_ST { 1 } else { 0 }
}
/// C StateRightDelta(sttype) (cm.c): 1 if the state emits right (MP/MR/IR), else 0.
#[inline]
fn sdr_of(stt: i32) -> i32 {
    if stt == MP_ST || stt == MR_ST || stt == IR_ST { 1 } else { 0 }
}

/// C alphabet.c:DegeneratePairScore(abc, esc, syml, symr). Canonical pair is a
/// direct lookup; gap/missing gets IMPOSSIBLE; otherwise the degeneracy-weighted
/// average `sum_l sum_r esc[l*K+r]*left[l]*right[r]` (left/right = FCount fractions).
fn degenerate_pair_score(esc: &[f32], syml: usize, symr: usize) -> f32 {
    const K: usize = ALPHABET_SIZE;
    const KP: usize = ALPHABET_SIZE_P;
    if syml < K && symr < K {
        return esc[syml * K + symr];
    }
    if syml == K || symr == K {
        return crate::constants::IMPOSSIBLE_F32; // gap
    }
    if syml == KP - 1 || symr == KP - 1 {
        return crate::constants::IMPOSSIBLE_F32; // missing data
    }
    let mut sc = 0.0f32;
    for l in 0..K {
        for r in 0..K {
            sc += esc[l * K + r] * abc_fcount_frac(syml, l) * abc_fcount_frac(symr, r);
        }
    }
    sc
}

/// C reads `cm->esc[v][off]` where each `esc[v] = esc[0] + v*(K*K)` is a slice into
/// one contiguous `[nstates][K*K]` block. For a degenerate MP emission in Joint
/// mode, C's `struct_sc += esc[v][symi*K+symj]` with `symi >= K` indexes past the
/// 16-entry state block into a LATER state's block — a latent C bug (deterministic,
/// no crash, because the memory is one allocation). We reproduce that exact
/// contiguous read (Rust's per-state Vecs would otherwise panic out of bounds).
#[inline]
fn esc_contig(cm: &CM, v: usize, off: usize) -> f32 {
    let flat = v * (ALPHABET_SIZE * ALPHABET_SIZE) + off; // == esc[0] + v*16 + off
    let sv = flat / (ALPHABET_SIZE * ALPHABET_SIZE);
    let si = flat % (ALPHABET_SIZE * ALPHABET_SIZE);
    cm.esc.get(sv).and_then(|row| row.get(si)).copied().unwrap_or(0.0)
}

/// C: Statetype(sttype) (cm.c:2490) — short state-type name for ParsetreeDump.
fn statetype(stt: i32) -> &'static str {
    use crate::constants::{D_ST, EL_ST, S_ST};
    match stt {
        x if x == D_ST => "D",
        x if x == MP_ST => "MP",
        x if x == ML_ST => "ML",
        x if x == MR_ST => "MR",
        x if x == IL_ST => "IL",
        x if x == IR_ST => "IR",
        x if x == S_ST => "S",
        x if x == E_ST => "E",
        x if x == B_ST => "B",
        x if x == EL_ST => "EL",
        _ => "?",
    }
}

/// C: MarginalMode(mode) (cm.c) — marginal-mode name for ParsetreeDump.
fn marginal_mode(mode: i8) -> &'static str {
    match mode {
        x if x == TRMODE_J => "Joint",
        x if x == TRMODE_L => "Left",
        x if x == TRMODE_R => "Right",
        0 => "Term", // TRMODE_T
        _ => "Unkwn",
    }
}

/// C: ParsetreeScore(cm, emap=NULL, tr, dsq, do_null2=FALSE, &sc, &struct_sc, ...)
/// (cm_parsetree.c:445). Returns `(sc, struct_sc)` in bits. Faithful marginal-mode
/// port: truncated parses (`!tr.is_std`) add `tr.trpenalty` at the ROOT_S
/// transition, MP emissions use lmesc/rmesc for the L/R marginal cases.
pub fn parsetree_score(cm: &CM, tr: &Parsetree, dsq: &[u8]) -> (f32, f32) {
    let k = ALPHABET_SIZE;
    let m = cm.m;
    let mut sc = 0.0f32;
    let mut struct_sc = 0.0f32;
    for tidx in 0..tr.n as usize {
        let v = tr.state[tidx];
        let mode = tr.mode[tidx];
        if v == m {
            continue; // EL, local alignment end
        }
        let vu = v as usize;
        let stt = cm.sttype[vu] as i32;
        if stt != E_ST && stt != B_ST {
            // transition score contribution
            if tr.nxtl[tidx] == -1 {
                // truncated end: no transition score contribution
            } else {
                let y = tr.state[tr.nxtl[tidx] as usize];
                if v == 0 {
                    if !tr.is_std {
                        sc += tr.trpenalty;
                    } else if (cm.flags & CMH_LOCAL_BEGIN) != 0 {
                        sc += cm.beginsc[y as usize];
                    } else {
                        sc += cm.tsc[vu][(y - cm.cfirst[vu]) as usize];
                    }
                } else if y == m {
                    let sd = if mode == TRMODE_J {
                        state_delta_of(stt)
                    } else if mode == TRMODE_L {
                        sdl_of(stt)
                    } else if mode == TRMODE_R {
                        sdr_of(stt)
                    } else {
                        0
                    };
                    sc += cm.endsc[vu]
                        + cm.el_selfsc * (tr.emitr[tidx] - tr.emitl[tidx] + 1 - sd) as f32;
                } else {
                    sc += cm.tsc[vu][(y - cm.cfirst[vu]) as usize];
                }
            }
            // emission score contribution
            if stt == MP_ST {
                let symi = dsq[tr.emitl[tidx] as usize] as usize;
                let symj = dsq[tr.emitr[tidx] as usize] as usize;
                if mode == TRMODE_J {
                    if symi < k && symj < k {
                        sc += cm.esc[vu][symi * k + symj];
                        struct_sc += cm.esc[vu][symi * k + symj];
                    } else {
                        sc += degenerate_pair_score(&cm.esc[vu], symi, symj);
                        // C latent bug: contiguous esc read past the state block.
                        struct_sc += esc_contig(cm, vu, symi * k + symj);
                    }
                    let lsc = cm.lmesc[vu][symi];
                    let rsc = cm.rmesc[vu][symj];
                    struct_sc -= lsc;
                    struct_sc -= rsc;
                } else if mode == TRMODE_L {
                    sc += cm.lmesc[vu][symi];
                } else if mode == TRMODE_R {
                    sc += cm.rmesc[vu][symj];
                }
            } else if (stt == ML_ST || stt == IL_ST) && (mode == TRMODE_J || mode == TRMODE_L) {
                let symi = dsq[tr.emitl[tidx] as usize] as usize;
                let lsc = if symi < k {
                    cm.esc[vu][symi]
                } else {
                    abc_favg_score(symi, &cm.esc[vu])
                };
                sc += lsc;
            } else if (stt == MR_ST || stt == IR_ST) && (mode == TRMODE_J || mode == TRMODE_R) {
                let symj = dsq[tr.emitr[tidx] as usize] as usize;
                let rsc = if symj < k {
                    cm.esc[vu][symj]
                } else {
                    abc_favg_score(symj, &cm.esc[vu])
                };
                sc += rsc;
            }
        }
    }
    (sc, struct_sc)
}

/// C: ParsetreeDump(fp, tr, cm, dsq) (cm_parsetree.c:ParsetreeDump). Truncation-
/// aware: the header shows tr.is_std/pass_idx/trpenalty and the per-node esc/tsc
/// use marginal (L/R) scores for truncated parses.
pub fn parsetree_dump(
    out: &mut dyn std::io::Write,
    tr: &Parsetree,
    cm: &CM,
    dsq: &[u8],
) -> std::io::Result<()> {
    let m = cm.m;
    writeln!(out, "Parsetree dump")?;
    writeln!(out, "------------------")?;
    writeln!(
        out,
        "is_std              = {}",
        if tr.is_std {
            "TRUE (alignment is not truncated)"
        } else {
            "FALSE (parsetree was found by a truncated DP algorithm)"
        }
    )?;
    writeln!(out, "pass_idx            = {}", tr.pass_idx)?;
    writeln!(out, "trpenalty           = {:.3}", tr.trpenalty)?;
    writeln!(out, "parsetree:")?;
    writeln!(out)?;
    writeln!(
        out,
        "{:>5} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        " idx ", "emitl", "emitr", "state", " mode", " nxtl", " nxtr", " prv ", " tsc ", " esc "
    )?;
    writeln!(
        out,
        "{:>5} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "-----", "------", "------", "-------", "-----", "-----", "-----", "-----", "-----", "-----"
    )?;

    for x in 0..tr.n as usize {
        let v = tr.state[x];
        let mode = tr.mode[x];
        // st: EL for v==M (C reads cm->sttype[v] but EL nodes fall through to
        // ' '/0.; we branch on EL explicitly to avoid an out-of-range index).
        let el = v == m;
        let stt = if el { crate::constants::EL_ST } else { cm.sttype[v as usize] as i32 };

        // syml/symr/esc: only P/L/R states emit, gated by marginal mode.
        let mut syml = b' ';
        let mut symr = b' ';
        let mut esc = 0.0f32;
        let vu = v as usize;
        if stt == MP_ST {
            if mode == TRMODE_J || mode == TRMODE_L {
                syml = RNA_SYM[dsq[tr.emitl[x] as usize] as usize];
            }
            if mode == TRMODE_J || mode == TRMODE_R {
                symr = RNA_SYM[dsq[tr.emitr[x] as usize] as usize];
            }
            if mode == TRMODE_J {
                esc = degenerate_pair_score(
                    &cm.esc[vu],
                    dsq[tr.emitl[x] as usize] as usize,
                    dsq[tr.emitr[x] as usize] as usize,
                );
            } else if mode == TRMODE_L {
                esc = cm.lmesc[vu][dsq[tr.emitl[x] as usize] as usize];
            } else if mode == TRMODE_R {
                esc = cm.rmesc[vu][dsq[tr.emitr[x] as usize] as usize];
            }
        } else if (stt == IL_ST || stt == ML_ST) && (mode == TRMODE_J || mode == TRMODE_L) {
            syml = RNA_SYM[dsq[tr.emitl[x] as usize] as usize];
            esc = abc_favg_score(dsq[tr.emitl[x] as usize] as usize, &cm.esc[vu]);
        } else if (stt == IR_ST || stt == MR_ST) && (mode == TRMODE_J || mode == TRMODE_R) {
            symr = RNA_SYM[dsq[tr.emitr[x] as usize] as usize];
            esc = abc_favg_score(dsq[tr.emitr[x] as usize] as usize, &cm.esc[vu]);
        }

        // tsc: B/E/EL have no transitions.
        let mut tsc = 0.0f32;
        if !el && stt != B_ST && stt != E_ST && tr.nxtl[x] != -1 {
            let y = tr.state[tr.nxtl[x] as usize];
            if v == 0 {
                if !tr.is_std {
                    tsc = tr.trpenalty;
                } else if (cm.flags & CMH_LOCAL_BEGIN) != 0 {
                    tsc = cm.beginsc[y as usize];
                } else {
                    tsc = cm.tsc[vu][(y - cm.cfirst[vu]) as usize];
                }
            } else if y == m {
                let sd = if mode == TRMODE_J {
                    state_delta_of(stt)
                } else if mode == TRMODE_L {
                    sdl_of(stt)
                } else if mode == TRMODE_R {
                    sdr_of(stt)
                } else {
                    0
                };
                tsc = cm.endsc[vu] + cm.el_selfsc * (tr.emitr[x] - tr.emitl[x] + 1 - sd) as f32;
            } else {
                tsc = cm.tsc[vu][(y - cm.cfirst[vu]) as usize];
            }
        }

        // C: "%5d %5d%c %5d%c %5d%-2s %5s %5d %5d %5d %5.2f %5.2f\n"
        writeln!(
            out,
            "{:5} {:5}{} {:5}{} {:5}{:<2} {:>5} {:5} {:5} {:5} {:5.2} {:5.2}",
            x,
            tr.emitl[x],
            syml as char,
            tr.emitr[x],
            symr as char,
            tr.state[x],
            statetype(stt),
            marginal_mode(tr.mode[x]),
            tr.nxtl[x],
            tr.nxtr[x],
            tr.prv[x],
            tsc,
            esc,
        )?;
    }
    writeln!(
        out,
        "{:>5} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "-----", "------", "------", "-------", "-----", "-----", "-----", "-----", "-----", "-----"
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsetree_new() {
        let pt = Parsetree::new(100);
        assert_eq!(pt.n, 0);
        assert_eq!(pt.state.capacity(), 100);
    }

    #[test]
    fn test_parsetree_add_node() {
        let mut pt = Parsetree::new(10);

        // Add root node (state 0, no parent)
        let idx = pt.add_node(1, 76, 0, -1, -1, -1);
        assert_eq!(idx, 0);
        assert_eq!(pt.n, 1);
        assert_eq!(pt.state[0], 0);
        assert_eq!(pt.emitl[0], 1);
        assert_eq!(pt.emitr[0], 76);

        // Add child node
        let idx2 = pt.add_node(2, 75, 1, -1, -1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(pt.n, 2);
        assert_eq!(pt.prv[1], 0); // parent is node 0
    }

    #[test]
    fn test_parsetree_root() {
        let mut pt = Parsetree::new(10);
        assert_eq!(pt.root(), None);

        pt.add_node(1, 76, 0, -1, -1, -1);
        assert_eq!(pt.root(), Some(0));
    }

    #[test]
    fn test_parsetree_get_state() {
        let mut pt = Parsetree::new(10);
        pt.add_node(1, 76, 42, -1, -1, -1);

        assert_eq!(pt.get_state(0), Some(42));
        assert_eq!(pt.get_state(1), None);
    }

    #[test]
    fn test_parsetree_get_emissions() {
        let mut pt = Parsetree::new(10);
        pt.add_node(5, 10, 0, -1, -1, -1);

        assert_eq!(pt.get_emissions(0), Some((5, 10)));
        assert_eq!(pt.get_emissions(1), None);
    }
}
