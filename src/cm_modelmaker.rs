//! Faithful port of Infernal's CM construction pipeline (the core of `cmbuild`).
//!
//! Ports (with the C file:function noted at each block):
//!   - cm_modelmaker.c : HandModelmaker, AssignMatchColumnsForMsa, cm_from_guide,
//!                       Transmogrify, cm_find_and_detach_dual_inserts,
//!                       cm_detach_state, cm_check_before_detaching,
//!                       cm_zero_flanking_insert_counts, clean_cs
//!   - cm_parsetree.c  : CreateParsetree, InsertTraceNodewithMode, ParsetreeCount*
//!   - alphabet.c      : PairCount, esl_abc_FCount
//!   - esl_wuss.c      : esl_wuss2ct, esl_wuss_nopseudo
//!   - esl_msaweight.c : esl_msaweight_PB_adv (default: ignore_rf, no subsample)
//!   - esl_msa.c       : esl_msa_Checksum, esl_msa_MarkFragments
//!   - cmbuild.c       : flatten_insert_emissions, mark_fragments, and the
//!                       build_model / configure_model glue used by bin/cmbuild.rs
//!
//! Scope: the default `cmbuild` path (--wpb weighting, hand/auto matchcol,
//! non-local build). The p7 filter build and RIBOSUM/--rsearch are out of scope.

use crate::cm::CM;
use crate::constants::*;
use crate::easel::alphabet::EslAlphabet;
use crate::easel::msa::EslMsa;

// C: infernal.h
const TRACE_LEFT_CHILD: i32 = 1;
const TRACE_RIGHT_CHILD: i32 = 2;
const DUMMY_ND: i32 = -1;

// Digital alphabet codes (RNA): K=4, Kp=18.
const K: usize = 4;
const KP: usize = 18;
const GAP: u8 = 4; // esl gap symbol
const NONRESIDUE: u8 = 16; // '*'
const MISSING: u8 = 17; // '~'

#[inline]
fn xis_residue(x: u8) -> bool {
    (x as usize) < K || ((x as usize) > K && (x as usize) < KP - 2)
}
#[inline]
fn xis_gap(x: u8) -> bool {
    x == GAP
}
#[inline]
fn xis_missing(x: u8) -> bool {
    x == MISSING
}
#[inline]
fn xis_canonical(x: u8) -> bool {
    (x as usize) < K
}
#[inline]
fn cis_gap(c: u8) -> bool {
    c == b'.' || c == b'_' || c == b'-'
}
#[inline]
fn cis_missing(c: u8) -> bool {
    c == b'~'
}

// IUPAC RNA degeneracy: same table as cm.rs RNA_DEGEN, restated for [K].
#[inline]
fn degen(x: u8, y: usize) -> bool {
    crate::cm::RNA_DEGEN[x as usize][y]
}
#[inline]
fn ndegen(x: u8) -> usize {
    crate::cm::RNA_NDEGEN[x as usize]
}

// =============================================================================
// Build-time parse/guide tree (C: Parsetree_t, cm_parsetree.c)
// =============================================================================

/// C: Parsetree_t. Used both as the guide tree (state[]=node type) and as a
/// per-sequence parse tree (state[]=CM state index).
#[derive(Clone)]
pub struct Ptree {
    pub n: i32,
    pub emitl: Vec<i32>,
    pub emitr: Vec<i32>,
    pub state: Vec<i32>,
    pub mode: Vec<i8>,
    pub nxtl: Vec<i32>,
    pub nxtr: Vec<i32>,
    pub prv: Vec<i32>,
}

impl Ptree {
    /// C: CreateParsetree()
    pub fn create() -> Self {
        Ptree {
            n: 0,
            emitl: Vec::new(),
            emitr: Vec::new(),
            state: Vec::new(),
            mode: Vec::new(),
            nxtl: Vec::new(),
            nxtr: Vec::new(),
            prv: Vec::new(),
        }
    }

    /// C: InsertTraceNodewithMode(). Insert a node attached to node y (left or
    /// right child). y == -1 initializes the root.
    pub fn insert(&mut self, y: i32, whichway: i32, emitl: i32, emitr: i32, state: i32, mode: i8) -> i32 {
        let n = self.n;
        let a = if y >= 0 {
            if whichway == TRACE_LEFT_CHILD {
                self.nxtl[y as usize]
            } else {
                self.nxtr[y as usize]
            }
        } else {
            -1
        };
        self.emitl.push(emitl);
        self.emitr.push(emitr);
        self.state.push(state);
        self.mode.push(mode);
        self.nxtl.push(a);
        self.nxtr.push(-1);
        self.prv.push(y);
        if y >= 0 {
            if whichway == TRACE_LEFT_CHILD {
                self.nxtl[y as usize] = n;
            } else {
                self.nxtr[y as usize] = n;
            }
        }
        if a != -1 {
            self.prv[a as usize] = n;
        }
        self.n += 1;
        n
    }
    #[inline]
    pub fn insert_j(&mut self, y: i32, whichway: i32, emitl: i32, emitr: i32, state: i32) -> i32 {
        self.insert(y, whichway, emitl, emitr, state, TRMODE_J as i8)
    }
}

const TRMODE_J: i32 = 3;

// =============================================================================
// esl_wuss.c
// =============================================================================

/// C: esl_wuss_nopseudo(ss1, ss2) — copy, replacing any alpha char with '.'.
pub fn wuss_nopseudo(ss: &mut [u8]) {
    for c in ss.iter_mut() {
        if c.is_ascii_alphabetic() {
            *c = b'.';
        }
    }
}

/// C: esl_wuss2ct(ss, len, ct). ct is 1..len; ct[i] = pairing partner of i or 0.
/// Returns Err on syntax error. `ss` is 0-based length `len`.
pub fn wuss2ct(ss: &[u8], len: usize) -> Result<Vec<i32>, ()> {
    let mut ct = vec![0i32; len + 1];
    // pda[0] main structure, pda[1..=26] pseudoknot levels
    let mut pda: Vec<Vec<i32>> = vec![Vec::new(); 27];
    for pos in 1..=len {
        let c = ss[pos - 1];
        if !c.is_ascii_graphic() && c != b' ' {
            return Err(());
        }
        match c {
            b'<' | b'(' | b'[' | b'{' => pda[0].push(pos as i32),
            b'>' | b')' | b']' | b'}' => {
                let pair = pda[0].pop().ok_or(())?;
                let open = ss[pair as usize - 1];
                let ok = (open == b'<' && c == b'>')
                    || (open == b'(' && c == b')')
                    || (open == b'[' && c == b']')
                    || (open == b'{' && c == b'}');
                if !ok {
                    return Err(());
                }
                ct[pos] = pair;
                ct[pair as usize] = pos as i32;
            }
            _ if c.is_ascii_uppercase() => {
                let i = (c - b'A' + 1) as usize;
                pda[i].push(pos as i32);
            }
            _ if c.is_ascii_lowercase() => {
                let i = (c - b'a' + 1) as usize;
                let pair = pda[i].pop().ok_or(())?;
                ct[pos] = pair;
                ct[pair as usize] = pos as i32;
            }
            b':' | b',' | b'_' | b'-' | b'.' | b'~' => {}
            _ => return Err(()),
        }
    }
    for stack in &pda {
        if !stack.is_empty() {
            return Err(());
        }
    }
    Ok(ct)
}

/// C: esl_ct2wuss(ct, n, ss) — convert a base-pair partner array `ct` (1..n,
/// `ct[i]` = partner of i or 0) into a WUSS secondary-structure string `ss` of
/// length `n`. Faithful port of esl_wuss.c:esl_ct2wuss(), including the
/// pseudoknot-resolution path (which is dead for the nested CM consensus
/// structure cmemit produces, but ported for correctness). Returns Err on the
/// C exception conditions (unfound partner / too many pseudoknots / pair count
/// mismatch).
pub fn ct2wuss(ct: &[i32], n: usize) -> Result<Vec<u8>, ()> {
    let mut rb = [-1i32; 26]; // right bound per pseudoknot index
    let mut pda: Vec<i32> = Vec::new(); // main structure stack
    let mut auxpk: Vec<i32> = Vec::new(); // pseudoknot aux stack
    let mut auxss: Vec<i32> = Vec::new(); // single-stranded aux stack
    let mut cct: Vec<i32> = ct[..=n].to_vec(); // modifiable copy (index 0..=n)

    // total number of basepairs
    let mut npairs = 0;
    for j in 1..=n {
        if ct[j] > 0 && (j as i32) < ct[j] {
            npairs += 1;
        }
    }
    let mut npairs_reached = 0;

    // init ss[] to single-stranded (':'); length n
    let mut ss = vec![b':'; n];

    for j in 1..=n {
        if cct[j] == 0 {
            pda.push(j as i32); // unpaired: push j
        } else if cct[j] > j as i32 {
            pda.push(j as i32); // left side of a bp: push j
        } else {
            // right side of a bp: find the left partner of j
            let mut found_partner = false;
            let mut nfaces = 0;
            let mut minface = -1;
            while let Some(i) = pda.pop() {
                if i < 0 {
                    // a face counter
                    nfaces += 1;
                    if i < minface {
                        minface = i;
                    }
                } else if cct[i as usize] == j as i32 {
                    // found the i,j pair
                    found_partner = true;
                    npairs_reached += 1;
                    if nfaces > 1 && minface > -4 {
                        minface -= 1;
                    }
                    match minface {
                        -1 => {
                            ss[i as usize - 1] = b'<';
                            ss[j - 1] = b'>';
                        }
                        -2 => {
                            ss[i as usize - 1] = b'(';
                            ss[j - 1] = b')';
                        }
                        -3 => {
                            ss[i as usize - 1] = b'[';
                            ss[j - 1] = b']';
                        }
                        -4 => {
                            ss[i as usize - 1] = b'{';
                            ss[j - 1] = b'}';
                        }
                        _ => return Err(()), // "no such face code"
                    }
                    pda.push(minface);
                    // label the single-stranded residues we set aside
                    while let Some(is) = auxss.pop() {
                        ss[is as usize - 1] = match nfaces {
                            0 => b'_',
                            1 => b'-',
                            _ => b',',
                        };
                    }
                    break;
                } else if cct[i as usize] == 0 {
                    // add to auxss only if originally single-stranded
                    if ct[i as usize] == 0 {
                        auxss.push(i);
                    }
                } else {
                    // i is paired, but not to j: pseudoknot
                    auxpk.push(i);
                }
            }
            if !found_partner {
                return Err(()); // cannot find left partner (likely a triplet)
            }
        }

        // resolve pseudoknots found along the way
        if !auxpk.is_empty() {
            let mut leftbound = cct[j];
            let mut rightbound = leftbound + 1;
            let mut xpk: i32 = -1;
            while let Some(i) = auxpk.pop() {
                let mut k = rightbound - 1;
                let mut hit_leftbound = false;
                while k > leftbound {
                    if cct[k as usize] == 0 {
                        k -= 1;
                        continue;
                    } else if cct[k as usize] > rightbound {
                        k -= 1;
                        continue;
                    } else if cct[k as usize] == i {
                        break; // i continues the given pseudoknot
                    } else {
                        k = leftbound; // a new pseudoknot
                        hit_leftbound = true;
                        break;
                    }
                }
                if k == leftbound {
                    let _ = hit_leftbound;
                    // a new pseudoknot
                    xpk += 1;
                    while i < rb[xpk as usize] {
                        xpk += 1;
                    }
                    leftbound = if rightbound < cct[i as usize] {
                        rightbound
                    } else {
                        cct[j]
                    };
                    rightbound = cct[i as usize];
                }
                npairs_reached += 1;
                if xpk + (b'a' as i32) <= (b'z' as i32) {
                    if cct[i as usize] > rb[xpk as usize] {
                        rb[xpk as usize] = cct[i as usize];
                    }
                    ss[i as usize - 1] = (xpk as u8) + b'A';
                    ss[cct[i as usize] as usize - 1] = (xpk as u8) + b'a';
                    let cti = cct[i as usize];
                    cct[i as usize] = 0;
                    cct[ct[i as usize] as usize] = 0;
                    let _ = cti;
                } else {
                    return Err(()); // not enough letters for all pseudoknots
                }
            }
        }
    }

    if npairs != npairs_reached {
        return Err(());
    }
    Ok(ss)
}

/// C: cm_modelmaker.c:clean_cs(). Validate SS_cons and strip pseudoknots/bad
/// chars if necessary. Modifies `cs` in place. Returns false on parse failure.
pub fn clean_cs(cs: &mut [u8], alen: usize) -> bool {
    if wuss2ct(cs, alen).is_err() {
        // C calls cm_Fail here
        return false;
    }
    let has_pseudoknots = cs[..alen].iter().any(|c| c.is_ascii_alphabetic());
    if !has_pseudoknots {
        return true;
    }
    for i in 0..alen {
        let c = cs[i];
        if b"{[(<".contains(&c) || b">)]}".contains(&c) || b":_-,.~".contains(&c) {
            // keep
        } else if has_pseudoknots && c.is_ascii_alphabetic() {
            cs[i] = b'.';
        } else {
            cs[i] = b'.';
        }
    }
    wuss2ct(cs, alen).is_ok()
}

// =============================================================================
// esl_msa.c : checksum, fragment marking
// =============================================================================

/// C: esl_msa_Checksum() (digital mode).
pub fn msa_checksum(msa: &EslMsa) -> u32 {
    let mut val: u32 = 0;
    let alen = msa.alen as usize;
    for i in 0..msa.nseq {
        for pos in 1..=alen {
            val = val.wrapping_add(msa.ax[i][pos] as u32);
            val = val.wrapping_add(val << 10);
            val ^= val >> 6;
        }
    }
    val = val.wrapping_add(val << 3);
    val ^= val >> 11;
    val = val.wrapping_add(val << 15);
    val
}

/// C: esl_msa_MarkFragments() + cmbuild.c:mark_fragments() default path.
/// Mark seqs whose aligned span < fragthresh*alen as fragments and replace
/// their terminal gaps with the missing symbol. Modifies msa.ax in place.
pub fn mark_fragments(msa: &mut EslMsa, fragthresh: f32) {
    let alen = msa.alen as usize;
    let minspan = (fragthresh * alen as f32).ceil() as i32;
    for idx in 0..msa.nseq {
        let mut lpos = 1i32;
        while lpos <= alen as i32 && !xis_residue(msa.ax[idx][lpos as usize]) {
            lpos += 1;
        }
        let mut rpos = alen as i32;
        while rpos >= 1 && !xis_residue(msa.ax[idx][rpos as usize]) {
            rpos -= 1;
        }
        let is_frag = (rpos - lpos + 1) < minspan;
        if is_frag {
            // replace terminal gaps with missing data
            for pos in 1..=alen {
                if xis_residue(msa.ax[idx][pos]) {
                    break;
                }
                msa.ax[idx][pos] = MISSING;
            }
            for pos in (1..=alen).rev() {
                if xis_residue(msa.ax[idx][pos]) {
                    break;
                }
                msa.ax[idx][pos] = MISSING;
            }
        }
    }
}

/// C: cmbuild.c:check_fragments() (cmbuild.c:1485-1563). Called instead of
/// mark_fragments() when `--fraggiven` is used: rather than inferring fragments
/// from aligned span, we trust the missing-data (`~`) annotation already in the
/// input MSA and only VALIDATE it. Does NOT modify `msa.ax`. For each aligned
/// seq it checks that: (a) either all or no characters before the first residue
/// are `~`; (b) either all or no characters after the final residue are `~`;
/// (c) there are no internal `~`. On any violation it returns Err(<the exact
/// cm_Fail message>); the caller emits it via cm_Fail (`\nError: ...`, exit 1).
///
/// C is not verbose in the default cmbuild path (cfg->be_verbose == FALSE), so
/// the "Checking fragments ... done" stopwatch lines are never printed.
pub fn check_fragments(msa: &EslMsa) -> Result<(), String> {
    // C 1502: msa is always digital here (read_all digitizes). No re-check needed.
    let alen = msa.alen as i32;
    // C 1508: for (idx = 0; idx < msa->nseq; idx++)
    for idx in 0..msa.nseq {
        // C 1509-1523: find first residue (spos) and count leading missing (lmiss).
        let mut spos = 0i32;
        let mut lmiss = 0i32;
        let mut lpos = 1i32;
        while lpos <= alen {
            let x = msa.ax[idx][lpos as usize];
            if xis_residue(x) {
                spos = lpos;
                break; // C: lpos = msa->alen+1 to break the loop
            } else if xis_missing(x) {
                lmiss += 1;
            }
            lpos += 1;
        }
        // C 1524: if(spos == 0) cm_Fail("Sequence %d has 0 residues", idx+1);
        if spos == 0 {
            return Err(format!("Sequence {} has 0 residues", idx + 1));
        }

        // C 1526-1537: find final residue (epos) and count trailing missing (rmiss).
        let mut epos = alen + 1;
        let mut rmiss = 0i32;
        let mut rpos = alen;
        while rpos >= 1 {
            let x = msa.ax[idx][rpos as usize];
            if xis_residue(x) {
                epos = rpos;
                break; // C: rpos = 0 to break the loop
            } else if xis_missing(x) {
                rmiss += 1;
            }
            rpos -= 1;
        }
        // C 1538: if(epos == (msa->alen+1)) cm_Fail("Sequence %d has 0 residues", idx+1);
        if epos == alen + 1 {
            return Err(format!("Sequence {} has 0 residues", idx + 1));
        }

        // C 1541-1543: 5'-end fragment consistency check.
        if lmiss > 0 && lmiss != (spos - 1) {
            return Err(format!(
                "Sequence {} seems to be a fragment but first residue is position {} and it only has {} ~ at 5' end (should be {})",
                idx + 1, spos, lmiss, spos - 1
            ));
        }
        // C 1544-1546: 3'-end fragment consistency check.
        if rmiss > 0 && rmiss != (alen - epos) {
            return Err(format!(
                "Sequence {} seems to be a fragment but final residue is position {} and it only has {} ~ at 3' end (should be {})",
                idx + 1, epos, rmiss, alen - epos
            ));
        }
        // C 1548-1552: no internal ~ between first and final residue.
        // NOTE: latent C BUG (cmbuild.c:1550): the format string has TWO %d
        // ("Sequence %d ... position %d") but cm_Fail is passed only ONE arg
        // (`lpos`). So the FIRST %d prints lpos and the SECOND %d reads
        // uninitialized stack (UB). We faithfully reproduce the first field
        // (= lpos, not idx+1), and for the second field emit lpos as well; the
        // C value there is non-portable stack garbage and cannot be byte-matched.
        let mut lp = spos;
        while lp <= epos {
            if xis_missing(msa.ax[idx][lp as usize]) {
                return Err(format!(
                    "Sequence {} has ~ at position {}, but these should only occur before first residue or after final residue",
                    lp, lp
                ));
            }
            lp += 1;
        }
    }
    Ok(())
}

// =============================================================================
// esl_msaweight.c : esl_msaweight_PB_adv (default cmbuild path: ignore_rf=TRUE,
// no subsampling since nseq is well below sampthresh=50000).
// =============================================================================

/// C: esl_vec_DSum() — Kahan (compensated) summation.
fn kahan_dsum(vec: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &v in vec {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// C: esl_msaweight_PB_adv(). `ignore_rf` mirrors `cfg->ignore_rf`: the default
/// cmbuild path passes TRUE (consensus by symfrac over all columns); cmbuild
/// `--hand` passes FALSE, so if the MSA has RF the consensus columns come from
/// the RF annotation (consensus_by_rf) and counts are collected only in them.
pub fn msaweight_pb(msa: &mut EslMsa, ignore_rf: bool) {
    let alen = msa.alen as usize;
    let nseq = msa.nseq;
    if nseq == 1 {
        msa.wgt[0] = 1.0;
        return;
    }
    let fragthresh = 0.5f32;
    let symfrac = 0.5f32;
    // ct[apos=0..alen][a=0..Kp-1]
    let mut ct = vec![vec![0i32; KP]; alen + 1];

    // Determine consensus columns early if we can (C esl_msaweight_PB_adv:206).
    // consensus_by_rf (esl_msaweight.c:272): non-gap RF columns, GAP test only
    // (missing '~' is NOT treated as gap here, unlike AssignMatchColumnsForMsa).
    let mut conscols: Vec<usize> = Vec::new();
    if !ignore_rf {
        if let Some(rf) = msa.rf.as_ref() {
            let rfb = rf.as_bytes();
            for apos in 1..=alen {
                if !cis_gap(rfb[apos - 1]) {
                    conscols.push(apos);
                }
            }
        }
    }
    let mut ncons = conscols.len();

    // collect_counts() (esl_msaweight.c:435): fragment rule per seq; if we
    // already have consensus columns, count only in them (within the span).
    let minspan = (fragthresh * alen as f32).ceil() as i32;
    for idx in 0..nseq {
        let mut lpos = 1i32;
        while lpos <= alen as i32 && !xis_residue(msa.ax[idx][lpos as usize]) {
            lpos += 1;
        }
        let mut rpos = alen as i32;
        while rpos >= 1 && !xis_residue(msa.ax[idx][rpos as usize]) {
            rpos -= 1;
        }
        if rpos - lpos + 1 >= minspan {
            lpos = 1;
            rpos = alen as i32;
        }
        if ncons > 0 {
            // conscols is ascending; stop once past rpos (C: conscols[j] <= rpos).
            for &apos in conscols.iter() {
                if apos as i32 > rpos {
                    break;
                }
                if (apos as i32) < lpos {
                    continue;
                }
                let a = msa.ax[idx][apos] as usize;
                ct[apos][a] += 1;
            }
        } else {
            for apos in lpos..=rpos {
                let a = msa.ax[idx][apos as usize] as usize;
                ct[apos as usize][a] += 1;
            }
        }
    }

    // If we still have no consensus columns, determine them now via symfrac
    // (consensus_by_all, esl_msaweight.c:401).
    if ncons == 0 {
        for apos in 1..=alen {
            let mut tot = 0i32;
            for a in 0..(KP - 2) {
                tot += ct[apos][a];
            }
            if tot > 0 && ((ct[apos][K] as f32) / (tot as f32)) < symfrac {
                conscols.push(apos);
            } else if tot == 0 {
                // C: division by zero -> NaN; NaN < symfrac is false, so not consensus.
                // (Guard to avoid a panic; result identical to C's float NaN compare.)
            }
        }
        ncons = conscols.len();
    }
    // Pathological: no consensus columns -> use them all.
    let conscols: Vec<usize> = if ncons == 0 {
        (1..=alen).collect()
    } else {
        conscols
    };
    let ncons = conscols.len();

    // r[j] = number of distinct canonical residues used in each consensus col.
    let mut r = vec![0i32; ncons];
    for j in 0..ncons {
        let apos = conscols[j];
        for a in 0..K {
            if ct[apos][a] > 0 {
                r[j] += 1;
            }
        }
    }

    // PB weight rule.
    for idx in 0..nseq {
        msa.wgt[idx] = 0.0;
        let mut rlen = 0i32;
        for j in 0..ncons {
            let apos = conscols[j];
            let a = msa.ax[idx][apos] as usize;
            if a >= K {
                // no contribution
            } else {
                msa.wgt[idx] += 1.0 / ((r[j] * ct[apos][a]) as f64);
                rlen += 1;
            }
        }
        if rlen > 0 {
            msa.wgt[idx] /= rlen as f64;
        }
    }
    // Normalize to sum to N. C: esl_vec_DNorm (uses Kahan esl_vec_DSum) then
    // esl_vec_DScale. Kahan summation is load-bearing for f32 count parity.
    let sum = kahan_dsum(&msa.wgt[..nseq]);
    if sum != 0.0 {
        for w in &mut msa.wgt[..nseq] {
            *w /= sum;
        }
    } else {
        for w in &mut msa.wgt[..nseq] {
            *w = 1.0 / nseq as f64;
        }
    }
    for w in &mut msa.wgt[..nseq] {
        *w *= nseq as f64;
    }
    msa.flags |= crate::easel::msa::ESL_MSA_HASWGTS;
}

// =============================================================================
// cm_modelmaker.c : AssignMatchColumnsForMsa
// =============================================================================

/// C: AssignMatchColumnsForMsa(). Returns matassign[1..alen] (index 0 unused).
fn assign_match_columns(msa: &EslMsa, use_rf: bool, use_wts: bool, symfrac: f32) -> Vec<i32> {
    let alen = msa.alen as usize;
    let mut matassign = vec![0i32; alen + 1];
    if use_rf {
        let rf = msa.rf.as_ref().expect("use_rf but no RF").as_bytes();
        for apos in 1..=alen {
            let c = rf[apos - 1];
            matassign[apos] = if cis_gap(c) || cis_missing(c) { 0 } else { 1 };
        }
    } else {
        for apos in 1..=alen {
            let mut r = 0.0f32;
            let mut totwgt = 0.0f32;
            for idx in 0..msa.nseq {
                let wgt = if use_wts { msa.wgt[idx] as f32 } else { 1.0 };
                let x = msa.ax[idx][apos];
                if xis_residue(x) {
                    r += wgt;
                    totwgt += wgt;
                } else if xis_gap(x) {
                    totwgt += wgt;
                } else if xis_missing(x) {
                    continue;
                }
            }
            matassign[apos] = if r > 0.0 && r / totwgt >= symfrac { 1 } else { 0 };
        }
    }
    matassign
}

// =============================================================================
// cm_modelmaker.c : HandModelmaker + cm_from_guide
// =============================================================================

/// C: HandModelmaker(). Build a (count-zeroed) CM and its guide tree from an MSA.
/// Returns (cm, guide_tree). `use_wts` must be false when `use_rf` is true.
pub fn hand_modelmaker(
    msa: &mut EslMsa,
    abc: &EslAlphabet,
    use_rf: bool,
    use_wts: bool,
    symfrac: f32,
) -> (CM, Ptree) {
    let alen = msa.alen as usize;

    // 1. match/insert assignments
    let matassign = assign_match_columns(msa, use_rf, use_wts, symfrac);

    // 2. EL assignments: all FALSE (never used when building a model).
    let elassign = vec![false; alen + 1];

    // 3. ct[] base-pair partners (1..alen). Remove pseudoknots in place first.
    let mut ss: Vec<u8> = msa.ss_cons.as_ref().unwrap().as_bytes().to_vec();
    // ss_cons may be shorter/longer; ensure length alen
    ss.resize(alen, b'.');
    wuss_nopseudo(&mut ss);
    // write back the pseudoknot-stripped ss_cons (C modifies msa->ss_cons)
    msa.ss_cons = Some(String::from_utf8_lossy(&ss).into_owned());
    let mut ct = wuss2ct(&ss, alen).expect("consensus structure inconsistent");

    // 4. Make ct consistent with matassign; build c2a_map / a2c_map.
    let mut clen = 1usize; // C starts clen=1 here (recomputed to 0 below for gtr)
    for apos in 1..=alen {
        if matassign[apos] == 0 {
            if ct[apos] != 0 {
                let p = ct[apos] as usize;
                ct[p] = 0;
            }
            ct[apos] = 0;
        } else {
            clen += 1;
        }
    }
    // clen here is (# match cols)+1; allocate maps of that size.
    let mut c2a_map = vec![0i32; clen + 1];
    let mut a2c_map = vec![0i32; alen + 1];
    let mut cpos = 1usize;
    for apos in 1..=alen {
        if matassign[apos] == 1 {
            a2c_map[apos] = cpos as i32;
            c2a_map[cpos] = apos as i32;
            cpos += 1;
        } else {
            a2c_map[apos] = 0;
        }
    }

    // 5. Construct guide tree (preorder). nstates/nnodes/clen recomputed.
    let mut nstates = 0i32;
    let mut nnodes = 0i32;
    let mut gtr = Ptree::create();
    let mut pda: Vec<i32> = Vec::new();
    let mut clen2 = 0i32;

    // push (v=-1, emitl=1, emitr=alen, ROOT_nd)
    pda.push(-1);
    pda.push(1);
    pda.push(alen as i32);
    pda.push(ROOT_ND);

    while let Some(typ) = pda.pop() {
        let j0 = pda.pop().unwrap();
        let i0 = pda.pop().unwrap();
        let vparent = pda.pop().unwrap();
        let mut i = i0;
        let mut j = j0;
        // use_el is FALSE => el_i=i, el_j=j
        let el_i = i;
        let el_j = j;

        if i > j {
            // END
            let _v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, END_ND);
            nstates += 1;
            nnodes += 1;
        } else if typ == ROOT_ND {
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, ROOT_ND);
            while i <= j {
                if matassign[i as usize] == 1 || elassign[i as usize] {
                    break;
                }
                i += 1;
            }
            while j >= i {
                if matassign[j as usize] == 1 || elassign[j as usize] {
                    break;
                }
                j -= 1;
            }
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 3;
            nnodes += 1;
        } else if typ == BEGL_ND {
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, BEGL_ND);
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 1;
            nnodes += 1;
        } else if typ == BEGR_ND {
            let v = gtr.insert(vparent, TRACE_RIGHT_CHILD, i, j, BEGR_ND, TRMODE_J as i8);
            while i <= j {
                if matassign[i as usize] == 1 || elassign[i as usize] {
                    break;
                }
                i += 1;
            }
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 2;
            nnodes += 1;
        } else if ct[i as usize] == 0 {
            // MATL
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, el_j, MATL_ND);
            i += 1;
            while i <= j {
                if matassign[i as usize] == 1 || elassign[i as usize] {
                    break;
                }
                i += 1;
            }
            pda.push(v);
            pda.push(i);
            pda.push(el_j);
            pda.push(DUMMY_ND);
            nstates += 3;
            nnodes += 1;
            clen2 += 1;
        } else if ct[j as usize] == 0 {
            // MATR
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, el_i, j, MATR_ND);
            j -= 1;
            while j >= i {
                if matassign[j as usize] == 1 || elassign[j as usize] {
                    break;
                }
                j -= 1;
            }
            pda.push(v);
            pda.push(el_i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 3;
            nnodes += 1;
            clen2 += 1;
        } else if ct[i as usize] == j {
            // MATP
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, MATP_ND);
            i += 1;
            while i <= j {
                if matassign[i as usize] == 1 || elassign[i as usize] {
                    break;
                }
                i += 1;
            }
            j -= 1;
            while j >= i {
                if matassign[j as usize] == 1 || elassign[j as usize] {
                    break;
                }
                j -= 1;
            }
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 6;
            nnodes += 1;
            clen2 += 2;
        } else {
            // BIFURC: choose best split point k.
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, BIF_ND);
            let i_cpos = a2c_map[i as usize];
            let j_cpos = a2c_map[j as usize];
            let mut bestk = ct[i as usize] + 1;
            let mut bestdiff = alen as i32 + 1;
            let mut k = ct[i as usize] + 1;
            while k <= ct[j as usize] {
                let mut kp = k;
                while a2c_map[kp as usize] == 0 {
                    kp += 1;
                }
                let k_cpos = a2c_map[kp as usize];
                let diff = ((k_cpos - i_cpos) - (j_cpos - k_cpos + 1)).abs();
                if diff < bestdiff {
                    bestdiff = diff;
                    bestk = k;
                }
                while ct[k as usize] == 0 {
                    k += 1;
                }
                // for (k = ct[k]+1)
                k = ct[k as usize] + 1;
            }
            // push right BEGIN first
            pda.push(v);
            pda.push(bestk);
            pda.push(j);
            pda.push(BEGR_ND);
            // then left BEGIN
            pda.push(v);
            pda.push(i);
            pda.push(bestk - 1);
            pda.push(BEGL_ND);
            nstates += 1;
            nnodes += 1;
        }
    }

    let final_clen = clen2;
    // Build CM from guide tree.
    let mut cm = CM::new(nstates, nnodes);
    cm_from_guide(&mut cm, &gtr);
    cm.cm_zero();
    cm.clen = final_clen;

    // cm->map (always) = c2a_map[1..clen]
    cm.map = vec![0i32; (final_clen + 1) as usize];
    for cpos in 0..=(final_clen as usize) {
        cm.map[cpos] = c2a_map[cpos];
    }
    cm.flags |= crate::cm::CM_MAP;

    // cm->rf, only if use_rf
    if use_rf {
        let rf = msa.rf.as_ref().unwrap().as_bytes();
        let mut rfvec = vec![0u8; (final_clen + 2) as usize];
        rfvec[0] = b' ';
        for cpos in 1..=(final_clen as usize) {
            rfvec[cpos] = rf[(c2a_map[cpos] - 1) as usize];
        }
        cm.rf = rfvec;
        cm.flags |= crate::cm::CM_RF;
    }

    let _ = abc;
    (cm, gtr)
}

/// C: cm_from_guide(). Fill CM structural info from the guide tree.
fn cm_from_guide(cm: &mut CM, gtr: &Ptree) {
    // child_count / parent_count indexed by node type
    // {BIF,MATP,MATL,MATR,BEGL,BEGR,ROOT,END}
    let child_count = [1, 4, 2, 2, 1, 1, 0, 1];
    let parent_count = [1, 6, 3, 3, 1, 2, 3, 0];

    let mut node = 0i32;
    let mut state = 0i32;
    let mut clen = 0i32;
    let mut pda: Vec<i32> = Vec::new();
    pda.push(0);

    while let Some(v) = pda.pop() {
        let vt = gtr.state[v as usize];
        if vt == BIF_ND {
            let prvnodetype = gtr.state[gtr.prv[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = BIF_ND as i8;

            cm.sttype[state as usize] = B_ST as i8;
            cm.ndidx[state as usize] = node;
            cm.stid[state as usize] = BIF_B as i8;
            cm.cfirst[state as usize] = state + 1;
            cm.cnum[state as usize] = -1;
            pda.push(state);
            cm.plast[state as usize] = state - 1;
            cm.pnum[state as usize] = parent_count[prvnodetype as usize];
            state += 1;
            node += 1;
            pda.push(gtr.nxtr[v as usize]);
            pda.push(gtr.nxtl[v as usize]);
        } else if vt == MATP_ND {
            let nxtnodetype = gtr.state[gtr.nxtl[v as usize] as usize];
            let prvnodetype = gtr.state[gtr.prv[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = MATP_ND as i8;
            clen += 2;
            // MP
            set_state(cm, state, MP_ST, node, MATP_MP, state + 4, 2 + child_count[nxtnodetype as usize], state - 1, parent_count[prvnodetype as usize]);
            state += 1;
            // ML
            set_state(cm, state, ML_ST, node, MATP_ML, state + 3, 2 + child_count[nxtnodetype as usize], state - 2, parent_count[prvnodetype as usize]);
            state += 1;
            // MR
            set_state(cm, state, MR_ST, node, MATP_MR, state + 2, 2 + child_count[nxtnodetype as usize], state - 3, parent_count[prvnodetype as usize]);
            state += 1;
            // D
            set_state(cm, state, D_ST, node, MATP_D, state + 1, 2 + child_count[nxtnodetype as usize], state - 4, parent_count[prvnodetype as usize]);
            state += 1;
            // IL
            set_state(cm, state, IL_ST, node, MATP_IL, state, 2 + child_count[nxtnodetype as usize], state, 5);
            state += 1;
            // IR
            set_state(cm, state, IR_ST, node, MATP_IR, state, 1 + child_count[nxtnodetype as usize], state, 6);
            state += 1;
            node += 1;
            pda.push(gtr.nxtl[v as usize]);
        } else if vt == MATL_ND {
            let nxtnodetype = gtr.state[gtr.nxtl[v as usize] as usize];
            let prvnodetype = gtr.state[gtr.prv[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = MATL_ND as i8;
            clen += 1;
            set_state(cm, state, ML_ST, node, MATL_ML, state + 2, 1 + child_count[nxtnodetype as usize], state - 1, parent_count[prvnodetype as usize]);
            state += 1;
            set_state(cm, state, D_ST, node, MATL_D, state + 1, 1 + child_count[nxtnodetype as usize], state - 2, parent_count[prvnodetype as usize]);
            state += 1;
            set_state(cm, state, IL_ST, node, MATL_IL, state, 1 + child_count[nxtnodetype as usize], state, 3);
            state += 1;
            node += 1;
            pda.push(gtr.nxtl[v as usize]);
        } else if vt == MATR_ND {
            let nxtnodetype = gtr.state[gtr.nxtl[v as usize] as usize];
            let prvnodetype = gtr.state[gtr.prv[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = MATR_ND as i8;
            clen += 1;
            set_state(cm, state, MR_ST, node, MATR_MR, state + 2, 1 + child_count[nxtnodetype as usize], state - 1, parent_count[prvnodetype as usize]);
            state += 1;
            set_state(cm, state, D_ST, node, MATR_D, state + 1, 1 + child_count[nxtnodetype as usize], state - 2, parent_count[prvnodetype as usize]);
            state += 1;
            set_state(cm, state, IR_ST, node, MATR_IR, state, 1 + child_count[nxtnodetype as usize], state, 3);
            state += 1;
            node += 1;
            pda.push(gtr.nxtl[v as usize]);
        } else if vt == BEGL_ND {
            let nxtnodetype = gtr.state[gtr.nxtl[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = BEGL_ND as i8;
            set_state(cm, state, S_ST, node, BEGL_S, state + 1, child_count[nxtnodetype as usize], state - 1, 1);
            state += 1;
            node += 1;
            pda.push(gtr.nxtl[v as usize]);
        } else if vt == BEGR_ND {
            let nxtnodetype = gtr.state[gtr.nxtl[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = BEGR_ND as i8;
            let bifparent = pda.pop().unwrap();
            cm.cnum[bifparent as usize] = state; // right-child idx for BIF
            set_state(cm, state, S_ST, node, BEGR_S, state + 1, 1 + child_count[nxtnodetype as usize], bifparent, 1);
            state += 1;
            set_state(cm, state, IL_ST, node, BEGR_IL, state, 1 + child_count[nxtnodetype as usize], state, 2);
            state += 1;
            node += 1;
            pda.push(gtr.nxtl[v as usize]);
        } else if vt == ROOT_ND {
            let nxtnodetype = gtr.state[gtr.nxtl[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = ROOT_ND as i8;
            set_state(cm, state, S_ST, node, ROOT_S, state + 1, 2 + child_count[nxtnodetype as usize], -1, 0);
            state += 1;
            set_state(cm, state, IL_ST, node, ROOT_IL, state, 2 + child_count[nxtnodetype as usize], state, 2);
            state += 1;
            set_state(cm, state, IR_ST, node, ROOT_IR, state, 1 + child_count[nxtnodetype as usize], state, 3);
            state += 1;
            node += 1;
            pda.push(gtr.nxtl[v as usize]);
        } else if vt == END_ND {
            let prvnodetype = gtr.state[gtr.prv[v as usize] as usize];
            cm.nodemap[node as usize] = state;
            cm.ndtype[node as usize] = END_ND as i8;
            set_state(cm, state, E_ST, node, END_E, -1, 0, state - 1, parent_count[prvnodetype as usize]);
            state += 1;
            node += 1;
        }
    }
    cm.m = state;
    cm.nodes = node;
    cm.clen = clen;
}

/// C: ConsensusModelmaker (cm_modelmaker.c). Build a (count-zeroed) sub-CM and
/// its guide tree from a consensus secondary-structure string `ss_cons` (WUSS,
/// 0..clen-1) covering `clen` consensus columns. Every position is a consensus
/// (match) column, so this is a simpler variant of [`hand_modelmaker`]'s guide
/// tree construction (no matassign gaps). `building_sub_model` (C
/// will_never_localize) only gates validity checks that our `cm_from_guide` does
/// not perform, so it is accepted but unused. Returns (cm, guide_tree).
pub fn consensus_modelmaker(
    abc: &EslAlphabet,
    ss_cons: &[u8],
    clen: i32,
    _building_sub_model: bool,
) -> (CM, Ptree) {
    // 1. ct[] base-pair partners (1..clen). Remove pseudoknots in place first
    //    (esl_wuss_nopseudo), then esl_wuss2ct.
    let mut ss: Vec<u8> = ss_cons.to_vec();
    wuss_nopseudo(&mut ss);
    let ct = wuss2ct(&ss, clen as usize).expect("consensus structure inconsistent");

    // 2. Construct guide tree (preorder), tracking nstates/nnodes/obs_clen.
    let mut nstates = 0i32;
    let mut nnodes = 0i32;
    let mut obs_clen = 0i32;
    let mut gtr = Ptree::create();
    let mut pda: Vec<i32> = Vec::new();

    // push (v=-1, emitl=1, emitr=clen, ROOT_nd)
    pda.push(-1);
    pda.push(1);
    pda.push(clen);
    pda.push(ROOT_ND);

    while let Some(typ) = pda.pop() {
        let j = pda.pop().unwrap();
        let i = pda.pop().unwrap();
        let vparent = pda.pop().unwrap();
        let mut i = i;
        let mut j = j;

        if i > j {
            // END
            let _v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, END_ND);
            nstates += 1;
            nnodes += 1;
        } else if typ == ROOT_ND {
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, ROOT_ND);
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 3; // ROOT_nd -> S, IL, IR
            nnodes += 1;
        } else if typ == BEGL_ND {
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, BEGL_ND);
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 1; // BEGL_nd -> S
            nnodes += 1;
        } else if typ == BEGR_ND {
            let v = gtr.insert_j(vparent, TRACE_RIGHT_CHILD, i, j, BEGR_ND);
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 2; // BEGR_nd -> S, IL
            nnodes += 1;
        } else if ct[i as usize] == 0 {
            // MATL (i unpaired)
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, MATL_ND);
            i += 1;
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 3; // MATL_nd -> ML, D, IL
            nnodes += 1;
            obs_clen += 1;
        } else if ct[j as usize] == 0 {
            // MATR (j unpaired)
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, MATR_ND);
            j -= 1;
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 3; // MATR_nd -> MR, D, IL
            nnodes += 1;
            obs_clen += 1;
        } else if ct[i as usize] == j {
            // MATP (i,j paired to each other)
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, MATP_ND);
            i += 1;
            j -= 1;
            pda.push(v);
            pda.push(i);
            pda.push(j);
            pda.push(DUMMY_ND);
            nstates += 6; // MATP_nd -> MP, ML, MR, D, IL, IR
            nnodes += 1;
            obs_clen += 2;
        } else {
            // BIFURC (i,j paired, but not to each other). Choose most balanced
            // split point k (raw consensus positions; C cm_modelmaker.c).
            let v = gtr.insert_j(vparent, TRACE_LEFT_CHILD, i, j, BIF_ND);
            let mut bestk = ct[i as usize] + 1;
            let mut bestdiff = clen + 1;
            let mut k = ct[i as usize] + 1;
            while k <= ct[j as usize] {
                let diff = ((k - i) - (j - k + 1)).abs();
                if diff < bestdiff {
                    bestdiff = diff;
                    bestk = k;
                }
                while ct[k as usize] == 0 {
                    k += 1;
                }
                k = ct[k as usize] + 1;
            }
            // push the right BEGIN node first, then the left BEGIN node
            pda.push(v);
            pda.push(bestk);
            pda.push(j);
            pda.push(BEGR_ND);
            pda.push(v);
            pda.push(i);
            pda.push(bestk - 1);
            pda.push(BEGL_ND);
            nstates += 1; // BIF_nd -> B
            nnodes += 1;
        }
    }
    if obs_clen != clen {
        panic!("consensus_modelmaker(): obs_clen {} != clen {}", obs_clen, clen);
    }

    // 3. Build the CM from the guide tree (CreateCM + cm_from_guide + CMZero).
    // map/rf stay unset (invalid) — copied from the mother elsewhere if needed.
    let mut cm = CM::new(nstates, nnodes);
    cm_from_guide(&mut cm, &gtr);
    cm.cm_zero();
    cm.clen = clen;
    let _ = abc;
    (cm, gtr)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn set_state(cm: &mut CM, state: i32, sttype: i32, node: i32, stid: i32, cfirst: i32, cnum: i32, plast: i32, pnum: i32) {
    let s = state as usize;
    cm.sttype[s] = sttype as i8;
    cm.ndidx[s] = node;
    cm.stid[s] = stid as i8;
    cm.cfirst[s] = cfirst;
    cm.cnum[s] = cnum;
    cm.plast[s] = plast;
    cm.pnum[s] = pnum;
}

// =============================================================================
// cm_modelmaker.c : Transmogrify (non-truncated build path)
// =============================================================================

/// C: Transmogrify(). Build a fake parsetree for aligned digital seq `ax`
/// (1..alen with sentinel at 0), given CM and guide tree. `used_el` is the
/// all-FALSE array cmbuild passes (no EL emissions); truncation is handled but
/// never triggers for full-length training sequences.
pub fn transmogrify(cm: &CM, gtr: &Ptree, ax: &[u8], used_el: &[bool], alen: usize) -> Ptree {
    let mut tr = Ptree::create();
    let mut pda: Vec<i32> = Vec::new();
    let mut ended = false;

    // truncation preprocessing
    let mut spos = 1i32;
    let mut epos = alen as i32;
    for apos in 1..=alen {
        if !xis_missing(ax[apos]) {
            break;
        }
        spos += 1;
    }
    for apos in (1..=alen).rev() {
        if !xis_missing(ax[apos]) {
            break;
        }
        epos -= 1;
    }
    let trunc_5p = spos > 1;
    let trunc_3p = epos < alen as i32;
    let do_trunc = trunc_5p || trunc_3p;

    // nxt_mi / nxt_el (used_el is non-NULL in cmbuild)
    let mut nxt_mi = vec![(alen + 1) as i32; alen + 1];
    let mut nxt_el = vec![(alen + 1) as i32; alen + 1];
    {
        let mut prv_mi = 0usize;
        let mut prv_el = 0usize;
        for apos in 1..=alen {
            if !xis_gap(ax[apos]) {
                if used_el[apos] {
                    for a2 in prv_el..apos {
                        nxt_el[a2] = apos as i32;
                    }
                    prv_el = apos;
                } else {
                    for a2 in prv_mi..apos {
                        nxt_mi[a2] = apos as i32;
                    }
                    prv_mi = apos;
                }
            }
        }
    }

    let mut trunc_mode = TRMODE_J;
    let mut used_a = vec![false; cm.nodes as usize];
    if do_trunc {
        let mut trunc_begin_node = -1i32;
        for node in 0..cm.nodes as usize {
            if gtr.emitl[node] <= spos && gtr.emitr[node] >= epos {
                trunc_begin_node = node as i32;
            }
        }
        trunc_mode = trunc_mode_for(gtr.emitl[trunc_begin_node as usize], gtr.emitr[trunc_begin_node as usize], spos, epos);
        let _ = trunc_begin_node;
    }

    let mut tidx = -1i32;
    for node in 0..cm.nodes as usize {
        if do_trunc {
            // skip logic
            let ntype = gtr.state[node];
            if node != 0 && (node as i32) < trunc_begin_node_of(gtr, cm, spos, epos) {
                continue;
            } else if ntype == BEGL_ND || ntype == BEGR_ND {
                if !used_a[gtr.prv[node] as usize] {
                    continue;
                }
            } else if ntype == END_ND {
                if spos > gtr.emitr[node] || epos < gtr.emitr[node] {
                    continue;
                }
            } else if (spos > gtr.emitl[node] && spos > gtr.emitr[node])
                || (epos < gtr.emitl[node] && epos < gtr.emitr[node])
            {
                continue;
            }
            used_a[node] = true;
        }

        let ntype = gtr.state[node];
        let el = &used_el;
        match ntype {
            x if x == ROOT_ND => {
                tidx = tr.insert(tidx, TRACE_LEFT_CHILD, gtr.emitl[node], gtr.emitr[node], 0,
                    if do_trunc { trunc_mode as i8 } else { TRMODE_J as i8 });
                let nxt = gtr.nxtl[node] as usize;
                let mut i = gtr.emitl[node];
                while i < gtr.emitl[nxt] {
                    if !xis_gap(ax[i as usize]) && !xis_missing(ax[i as usize]) && !el[i as usize] {
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, gtr.emitr[node], 1,
                            if do_trunc { trunc_mode_for(i, gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                    }
                    i += 1;
                }
                let mut j = gtr.emitr[node];
                while j > gtr.emitr[nxt] {
                    if !xis_gap(ax[j as usize]) && !xis_missing(ax[j as usize]) && !el[j as usize] {
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, j, 2,
                            if do_trunc { trunc_mode_for(i, j, spos, epos) as i8 } else { TRMODE_J as i8 });
                    }
                    j -= 1;
                }
            }
            x if x == BIF_ND => {
                if ended {
                    continue;
                }
                let state = cm.calculate_state_index(node, BIF_B);
                tidx = tr.insert(tidx, TRACE_LEFT_CHILD, gtr.emitl[node], gtr.emitr[node], state,
                    if do_trunc { trunc_mode_for(gtr.emitl[node], gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                pda.push(if ended { 1 } else { 0 });
                pda.push(tidx);
            }
            x if x == MATP_ND => {
                let el_gap = xis_gap(ax[gtr.emitl[node] as usize]);
                let er_gap = xis_gap(ax[gtr.emitr[node] as usize]);
                let typ = if el_gap {
                    if er_gap { MATP_D } else { MATP_MR }
                } else if er_gap {
                    MATP_ML
                } else {
                    MATP_MP
                };
                if typ == MATP_D && ended {
                    continue;
                }
                let mut state = cm.calculate_state_index(node, typ);
                tidx = tr.insert(tidx, TRACE_LEFT_CHILD, gtr.emitl[node], gtr.emitr[node], state,
                    if do_trunc { trunc_mode_for(gtr.emitl[node], gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                // EL check (goto_el always FALSE for all-FALSE used_el) -- faithful.
                if typ == MATP_MP && cm.ndtype[node + 1] as i32 != END_ND {
                    if let Some((i, j)) = check_for_el(ax, el, &nxt_mi, &nxt_el, gtr.emitl[node] + 1, gtr.emitr[node] - 1) {
                        state = cm.m;
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, j, state,
                            if do_trunc { trunc_mode_for(i, j, spos, epos) as i8 } else { TRMODE_J as i8 });
                        ended = true;
                    }
                }
                if ended {
                    continue;
                }
                let nxt = gtr.nxtl[node] as usize;
                state = cm.calculate_state_index(node, MATP_IL);
                let mut i = gtr.emitl[node] + 1;
                while i < gtr.emitl[nxt] {
                    if !xis_gap(ax[i as usize]) && !xis_missing(ax[i as usize]) && !el[i as usize] {
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, gtr.emitr[node] - 1, state,
                            if do_trunc { trunc_mode_for(i, gtr.emitr[node] - 1, spos, epos) as i8 } else { TRMODE_J as i8 });
                    }
                    i += 1;
                }
                state = cm.calculate_state_index(node, MATP_IR);
                let mut j = gtr.emitr[node] - 1;
                while j > gtr.emitr[nxt] {
                    if !xis_gap(ax[j as usize]) && !xis_missing(ax[j as usize]) && !el[j as usize] {
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, j, state,
                            if do_trunc { trunc_mode_for(i, j, spos, epos) as i8 } else { TRMODE_J as i8 });
                    }
                    j -= 1;
                }
            }
            x if x == MATL_ND => {
                let typ = if xis_gap(ax[gtr.emitl[node] as usize]) { MATL_D } else { MATL_ML };
                if typ == MATL_D && ended {
                    continue;
                }
                let mut state = cm.calculate_state_index(node, typ);
                tidx = tr.insert(tidx, TRACE_LEFT_CHILD, gtr.emitl[node], gtr.emitr[node], state,
                    if do_trunc { trunc_mode_for(gtr.emitl[node], gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                if typ == MATL_ML && cm.ndtype[node + 1] as i32 != END_ND {
                    if let Some((i, j)) = check_for_el(ax, el, &nxt_mi, &nxt_el, gtr.emitl[node] + 1, gtr.emitr[node]) {
                        state = cm.m;
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, j, state,
                            if do_trunc { trunc_mode_for(i, j, spos, epos) as i8 } else { TRMODE_J as i8 });
                        ended = true;
                    }
                }
                if ended {
                    continue;
                }
                let nxt = gtr.nxtl[node] as usize;
                state = cm.calculate_state_index(node, MATL_IL);
                let mut i = gtr.emitl[node] + 1;
                while i < gtr.emitl[nxt] {
                    if !xis_gap(ax[i as usize]) && !xis_missing(ax[i as usize]) && !el[i as usize] {
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, gtr.emitr[node], state,
                            if do_trunc { trunc_mode_for(i, gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                    }
                    i += 1;
                }
            }
            x if x == MATR_ND => {
                let typ = if xis_gap(ax[gtr.emitr[node] as usize]) { MATR_D } else { MATR_MR };
                if typ == MATR_D && ended {
                    continue;
                }
                let mut state = cm.calculate_state_index(node, typ);
                tidx = tr.insert(tidx, TRACE_LEFT_CHILD, gtr.emitl[node], gtr.emitr[node], state,
                    if do_trunc { trunc_mode_for(gtr.emitl[node], gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                if typ == MATR_MR && cm.ndtype[node + 1] as i32 != END_ND {
                    if let Some((i, j)) = check_for_el(ax, el, &nxt_mi, &nxt_el, gtr.emitl[node], gtr.emitr[node] - 1) {
                        state = cm.m;
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, j, state,
                            if do_trunc { trunc_mode_for(i, j, spos, epos) as i8 } else { TRMODE_J as i8 });
                        ended = true;
                    }
                }
                if ended {
                    continue;
                }
                let nxt = gtr.nxtl[node] as usize;
                state = cm.calculate_state_index(node, MATR_IR);
                let mut j = gtr.emitr[node] - 1;
                while j > gtr.emitr[nxt] {
                    if !xis_gap(ax[j as usize]) && !xis_missing(ax[j as usize]) && !el[j as usize] {
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, gtr.emitl[node], j, state,
                            if do_trunc { trunc_mode_for(gtr.emitl[node], j, spos, epos) as i8 } else { TRMODE_J as i8 });
                    }
                    j -= 1;
                }
            }
            x if x == BEGL_ND => {
                if ended {
                    continue;
                }
                let mut state = cm.calculate_state_index(node, BEGL_S);
                tidx = tr.insert(tidx, TRACE_LEFT_CHILD, gtr.emitl[node], gtr.emitr[node], state,
                    if do_trunc { trunc_mode_for(gtr.emitl[node], gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                if cm.ndtype[node + 1] as i32 != END_ND {
                    if let Some((i, j)) = check_for_el(ax, el, &nxt_mi, &nxt_el, gtr.emitl[node], gtr.emitr[node]) {
                        state = cm.m;
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, j, state,
                            if do_trunc { trunc_mode_for(i, j, spos, epos) as i8 } else { TRMODE_J as i8 });
                        ended = true;
                    }
                }
            }
            x if x == BEGR_ND => {
                let popped_tidx = pda.pop().unwrap();
                let popped_ended = pda.pop().unwrap();
                tidx = popped_tidx;
                ended = popped_ended != 0;
                if ended {
                    continue;
                }
                let mut state = cm.calculate_state_index(node, BEGR_S);
                tidx = tr.insert(tidx, TRACE_RIGHT_CHILD, gtr.emitl[node], gtr.emitr[node], state,
                    if do_trunc { trunc_mode_for(gtr.emitl[node], gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                if cm.ndtype[node + 1] as i32 != END_ND {
                    if let Some((i, j)) = check_for_el(ax, el, &nxt_mi, &nxt_el, gtr.emitl[node], gtr.emitr[node]) {
                        state = cm.m;
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, j, state,
                            if do_trunc { trunc_mode_for(i, j, spos, epos) as i8 } else { TRMODE_J as i8 });
                        ended = true;
                    }
                }
                if ended {
                    continue;
                }
                let nxt = gtr.nxtl[node] as usize;
                state = cm.calculate_state_index(node, BEGR_IL);
                let mut i = gtr.emitl[node];
                while i < gtr.emitl[nxt] {
                    if !xis_gap(ax[i as usize]) && !xis_missing(ax[i as usize]) && !el[i as usize] {
                        tidx = tr.insert(tidx, TRACE_LEFT_CHILD, i, gtr.emitr[node], state,
                            if do_trunc { trunc_mode_for(i, gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                    }
                    i += 1;
                }
            }
            x if x == END_ND => {
                if !ended {
                    let state = cm.calculate_state_index(node, END_E);
                    tidx = tr.insert(tidx, TRACE_LEFT_CHILD, -1, -1, state,
                        if do_trunc { trunc_mode_for(gtr.emitl[node], gtr.emitr[node], spos, epos) as i8 } else { TRMODE_J as i8 });
                }
            }
            _ => panic!("Transmogrify: bogus node type"),
        }
    }
    tr
}

fn trunc_begin_node_of(gtr: &Ptree, cm: &CM, spos: i32, epos: i32) -> i32 {
    let mut best = -1i32;
    for node in 0..cm.nodes as usize {
        if gtr.emitl[node] <= spos && gtr.emitr[node] >= epos {
            best = node as i32;
        }
    }
    best
}

/// C: trunc_mode_for_trace_node()
fn trunc_mode_for(emitl: i32, emitr: i32, spos: i32, epos: i32) -> i32 {
    if emitl >= spos && emitr <= epos {
        3 // TRMODE_J
    } else if emitl >= spos && emitr > epos {
        2 // TRMODE_L
    } else if emitl < spos && emitr <= epos {
        1 // TRMODE_R
    } else {
        0 // TRMODE_T
    }
}

/// C: check_for_el(). Returns Some((i,j)) if we should transit to EL. For the
/// all-FALSE used_el array cmbuild passes, this always returns None.
fn check_for_el(ax: &[u8], used_el: &[bool], nxt_mi: &[i32], nxt_el: &[i32], i0: i32, j0: i32) -> Option<(i32, i32)> {
    if nxt_mi[(i0 - 1) as usize] > nxt_el[(i0 - 1) as usize]
        && nxt_mi[(i0 - 1) as usize] > j0
        && nxt_el[(i0 - 1) as usize] <= j0
    {
        let mut i = i0;
        let mut j = j0;
        while xis_gap(ax[i as usize]) && i <= j0 {
            i += 1;
        }
        while xis_gap(ax[j as usize]) && j >= i0 {
            j -= 1;
        }
        if i > j0 || !used_el[i as usize] {
            return None;
        }
        if j < i0 || !used_el[j as usize] {
            return None;
        }
        let mut i2 = i;
        while i2 <= j {
            if !used_el[i2 as usize] {
                break;
            }
            i2 += 1;
        }
        if i2 == j + 1 {
            return Some((i, j));
        }
    }
    None
}

// =============================================================================
// cm_parsetree.c : counting; alphabet.c : PairCount / FCount
// =============================================================================

/// C: esl_abc_FCount() into a K-length counter (residue/degenerate only).
fn fcount(ct: &mut [f32], x: u8, wt: f32) {
    if xis_canonical(x) {
        ct[x as usize] += wt;
    } else if xis_gap(x) {
        // C would add to ct[gap]; not reachable for emissions here.
    } else if xis_missing(x) || x == NONRESIDUE {
        // nothing
    } else {
        let nd = ndegen(x) as f32;
        for y in 0..K {
            if degen(x, y) {
                ct[y] += wt / nd;
            }
        }
    }
}

/// C: PairCount() into a K*K counter.
fn pair_count(counters: &mut [f32], syml: u8, symr: u8, wt: f32) {
    if xis_canonical(syml) && xis_canonical(symr) {
        counters[(syml as usize) * K + (symr as usize)] += wt;
    } else {
        let mut left = [0.0f32; K];
        let mut right = [0.0f32; K];
        fcount(&mut left, syml, 1.0);
        fcount(&mut right, symr, 1.0);
        for l in 0..K {
            for r in 0..K {
                counters[l * K + r] += left[l] * right[r] * wt;
            }
        }
    }
}

/// C: cm_parsetree.c:NumReachableInserts()
fn num_reachable_inserts(stid: i32) -> usize {
    match stid {
        MATL_ML | MATL_D | MATL_IL => 1,
        MATP_MP | MATP_ML | MATP_MR | MATP_D | MATP_IL => 2,
        MATP_IR => 1,
        MATR_MR | MATR_D | MATR_IR => 1,
        BIF_B | BEGL_S => 0,
        BEGR_S | BEGR_IL => 1,
        END_E => 0,
        ROOT_S | ROOT_IL => 2,
        ROOT_IR => 1,
        EL => 0,
        _ => panic!("bogus stid {}", stid),
    }
}

/// C: ParsetreeCountExceptTruncatedMPs(). Counts transitions and (mode-aware)
/// emissions into the count-based CM.
pub fn parsetree_count(cm: &mut CM, tr: &Ptree, dsq: &[u8], wgt: f32) {
    for tidx in 0..tr.n as usize {
        let v = tr.state[tidx];
        let mode = tr.mode[tidx] as i32;
        if v != cm.m
            && cm.sttype[v as usize] as i32 != E_ST
            && cm.sttype[v as usize] as i32 != B_ST
            && (mode == TRMODE_J || tr.nxtl[tidx] != -1)
        {
            let vu = v as usize;
            // transition
            if tidx < (tr.n as usize - 1) {
                let z = tr.state[tr.nxtl[tidx] as usize];
                if z == cm.m {
                    cm.end[vu] += wgt;
                } else if v == 0 && z - cm.cfirst[vu] >= cm.cnum[vu] {
                    cm.begin[z as usize] += wgt;
                } else {
                    cm.t[vu][(z - cm.cfirst[vu]) as usize] += wgt;
                }
            }
            // emission (mode-aware)
            let st = cm.sttype[vu] as i32;
            if mode == TRMODE_J {
                if st == MP_ST {
                    pair_count(&mut cm.e[vu], dsq[tr.emitl[tidx] as usize], dsq[tr.emitr[tidx] as usize], wgt);
                } else if st == ML_ST || st == IL_ST {
                    fcount(&mut cm.e[vu], dsq[tr.emitl[tidx] as usize], wgt);
                } else if st == MR_ST || st == IR_ST {
                    fcount(&mut cm.e[vu], dsq[tr.emitr[tidx] as usize], wgt);
                }
            } else if mode == 2 {
                // TRMODE_L
                if st == ML_ST || st == IL_ST {
                    fcount(&mut cm.e[vu], dsq[tr.emitl[tidx] as usize], wgt);
                }
            } else if mode == 1 {
                // TRMODE_R
                if st == MR_ST || st == IR_ST {
                    fcount(&mut cm.e[vu], dsq[tr.emitr[tidx] as usize], wgt);
                }
            }
        }
    }

    // Special-case IL/IR self-transition hack for identical final two nodes.
    let n = tr.n as usize;
    if n >= 2
        && tr.state[n - 2] == tr.state[n - 1]
        && tr.mode[n - 2] == tr.mode[n - 1]
    {
        let v = tr.state[n - 1];
        let vu = v as usize;
        let st = cm.sttype[vu] as i32;
        let m = tr.mode[n - 1] as i32;
        if (st == IL_ST && (m == TRMODE_J || m == 2)) || (st == IR_ST && (m == TRMODE_J || m == 1)) {
            let idx = num_reachable_inserts(cm.stid[vu] as i32);
            cm.t[vu][idx] += wgt;
        }
    }
}

/// C: cm_parsetree.c:cm_parsetree_Doctor(). For a "pretend CM is HMM" (0
/// basepair) model, rewrite a MATL-only parsetree to remove the D->I / I->D
/// ambiguity (each such pair becomes a single M state). Modifies `tr` in place.
pub fn cm_parsetree_doctor(cm: &CM, tr: &mut Ptree) {
    let n = tr.n as usize;
    // Determine truncation-mode case (1: all J, 2: all L, 3: R,R then J).
    macro_rules! stid {
        ($x:expr) => {
            cm.stid[tr.state[$x] as usize] as i32
        };
    }
    let mut mode_case = 1;
    for x in 0..n {
        if tr.mode[x] as i32 != TRMODE_J {
            mode_case = -1;
            break;
        }
    }
    if mode_case == -1 {
        mode_case = 2;
        for x in 0..n {
            if tr.mode[x] as i32 != 2 {
                mode_case = -1;
                break;
            }
        }
    }
    if mode_case == -1 {
        mode_case = 3;
        if tr.mode[0] as i32 == 1 && tr.mode[1] as i32 == 1 && stid!(0) == ROOT_S && stid!(1) == MATL_ML {
            for x in 2..n {
                if tr.mode[x] as i32 != TRMODE_J {
                    mode_case = -1;
                    break;
                }
            }
        } else {
            mode_case = -1;
        }
    }
    if mode_case == -1 {
        panic!("cm_parsetree_Doctor() unable to determine truncation mode case");
    }

    let (mode, start_node): (i32, usize) = match mode_case {
        1 => (TRMODE_J, 0),
        2 => (2, 0),
        _ => (TRMODE_J, 2),
    };

    // Rewrite left-to-right.
    let mut ndi = 0;
    let mut nid = 0;
    let mut opos = start_node;
    let mut npos = start_node;
    let _ = (ndi, nid);
    while opos < tr.n as usize {
        tr.mode[npos] = mode as i8;
        tr.nxtl[npos] = (npos + 1) as i32;
        tr.nxtr[npos] = -1;
        tr.prv[npos] = npos as i32 - 1;
        tr.emitr[npos] = tr.emitr[opos];

        let cur = stid!(opos);
        let nxt = if opos < tr.n as usize - 1 { stid!(opos + 1) } else { -1 };
        if opos < (tr.n as usize - 1) && cur == MATL_D && nxt == MATL_IL {
            tr.state[npos] = tr.state[opos] - 1; // MATL_D -> MATL_ML
            tr.emitl[npos] = tr.emitl[opos + 1];
            opos += 2;
            npos += 1;
            ndi += 1;
        } else if opos < (tr.n as usize - 1) && cur == MATL_IL && nxt == MATL_D {
            tr.state[npos] = tr.state[opos + 1] - 1; // MATL_D -> MATL_ML
            tr.emitl[npos] = tr.emitl[opos];
            opos += 2;
            npos += 1;
            nid += 1;
        } else {
            tr.state[npos] = tr.state[opos];
            tr.emitl[npos] = tr.emitl[opos];
            opos += 1;
            npos += 1;
        }
    }
    tr.n = npos as i32;
    tr.nxtl[tr.n as usize - 1] = -1;
    if mode_case == 1 || mode_case == 3 {
        tr.emitr[tr.n as usize - 1] = -1;
    }
}

/// C: alphabet.c:PairCountMarginal(). Partition weight <wt> for a half-observed
/// (truncated) base pair into the K*K emission counters, using mean-posterior
/// Dirichlet probabilities from the already-collected full-pair counts
/// <nonmarg> (dbl_e[v]) and the base-pair prior <pri.mbp>.
fn pair_count_marginal(
    counters: &mut [f32],
    nonmarg: &[f64],
    syml: u8,
    symr: u8,
    wt: f32,
    pri: &crate::prior::Prior,
) {
    let mut probs = [0.0f64; K * K];
    let wtd = wt as f64;
    if xis_missing(symr) {
        // C: esl_mixdchlet_MPParameters(pri->mbp, nonmarg_counters, probs)
        let mut mbp = pri.mbp.clone();
        mbp.mp_parameters(nonmarg, &mut probs);
        if (syml as usize) < K {
            let sl = syml as usize;
            let mut sum = 0.0f64;
            for r in 0..K {
                sum += probs[sl * K + r];
            }
            for r in 0..K {
                counters[sl * K + r] =
                    (counters[sl * K + r] as f64 + (probs[sl * K + r] / sum) * wtd) as f32;
            }
        } else {
            let mut left = [0.0f32; K];
            fcount(&mut left, syml, 1.0);
            let mut sum = 0.0f64;
            for l in 0..K {
                for r in 0..K {
                    sum += probs[l * K + r] * left[l] as f64;
                }
            }
            for l in 0..K {
                for r in 0..K {
                    counters[l * K + r] = (counters[l * K + r] as f64
                        + (probs[l * K + r] * left[l] as f64 / sum) * wtd)
                        as f32;
                }
            }
        }
    } else if xis_missing(syml) {
        let mut mbp = pri.mbp.clone();
        mbp.mp_parameters(nonmarg, &mut probs);
        if (symr as usize) < K {
            let sr = symr as usize;
            let mut sum = 0.0f64;
            for l in 0..K {
                sum += probs[l * K + sr];
            }
            for l in 0..K {
                counters[l * K + sr] =
                    (counters[l * K + sr] as f64 + (probs[l * K + sr] / sum) * wtd) as f32;
            }
        } else {
            let mut right = [0.0f32; K];
            fcount(&mut right, symr, 1.0);
            let mut sum = 0.0f64;
            for l in 0..K {
                for r in 0..K {
                    sum += probs[l * K + r] * right[r] as f64;
                }
            }
            for l in 0..K {
                for r in 0..K {
                    counters[l * K + r] = (counters[l * K + r] as f64
                        + (probs[l * K + r] * right[r] as f64 / sum) * wtd)
                        as f32;
                }
            }
        }
    } else {
        panic!("pair_count_marginal: neither syml nor symr missing");
    }
}

/// C: ParsetreeCountOnlyTruncatedMPs(). Count the truncated (half-observed) MP
/// emissions using mean-posterior estimates over the frozen full-pair counts
/// <dbl_e>. Call once per parsetree AFTER parsetree_count() has run for all
/// sequences and <dbl_e> has snapshotted the MP counts.
pub fn parsetree_count_only_truncated_mps(
    cm: &mut CM,
    tr: &Ptree,
    dsq: &[u8],
    wgt: f32,
    dbl_e: &[Vec<f64>],
    pri: &crate::prior::Prior,
) {
    for tidx in 0..tr.n as usize {
        let v = tr.state[tidx];
        if v != cm.m && cm.sttype[v as usize] as i32 == MP_ST {
            let vu = v as usize;
            let mode = tr.mode[tidx] as i32;
            if mode == 2 {
                // TRMODE_L: right symbol missing
                let syml = dsq[tr.emitl[tidx] as usize];
                let nonmarg = dbl_e[vu].clone();
                pair_count_marginal(&mut cm.e[vu], &nonmarg, syml, MISSING, wgt, pri);
            } else if mode == 1 {
                // TRMODE_R: left symbol missing
                let symr = dsq[tr.emitr[tidx] as usize];
                let nonmarg = dbl_e[vu].clone();
                pair_count_marginal(&mut cm.e[vu], &nonmarg, MISSING, symr, wgt, pri);
            }
        }
    }
}

// =============================================================================
// cm_modelmaker.c : cm_zero_flanking_insert_counts, detach
// =============================================================================

/// C: cm_zero_flanking_insert_counts()
pub fn cm_zero_flanking_insert_counts(cm: &mut CM) {
    cm.t[0][0] = 0.0;
    cm.t[0][1] = 0.0;
    match cm.ndtype[1] as i32 {
        x if x == BIF_ND => {
            cm.t[0][2] += cm.t[1][2];
            cm.t[0][2] += cm.t[2][1];
        }
        x if x == MATP_ND => {
            cm.t[0][2] += cm.t[1][2];
            cm.t[0][2] += cm.t[2][1];
            cm.t[0][3] += cm.t[1][3];
            cm.t[0][3] += cm.t[2][2];
            cm.t[0][4] += cm.t[1][4];
            cm.t[0][4] += cm.t[2][3];
            cm.t[0][5] += cm.t[1][5];
            cm.t[0][5] += cm.t[2][4];
        }
        _ => {
            cm.t[0][2] += cm.t[1][2];
            cm.t[0][2] += cm.t[2][1];
            cm.t[0][3] += cm.t[1][3];
            cm.t[0][3] += cm.t[2][2];
        }
    }
    for x in 0..CM_MAXCONNECT_LOCAL {
        cm.t[1][x] = 0.0;
        cm.t[2][x] = 0.0;
    }
}

const CM_MAXCONNECT_LOCAL: usize = 6;

/// C: cm_find_and_detach_dual_inserts(). do_check verifies END_E-1 inserts have
/// 0 counts; do_detach zeroes transitions into the detached insert.
pub fn cm_find_and_detach_dual_inserts(cm: &mut CM, do_check: bool, do_detach: bool) -> bool {
    let emap = crate::cm_emitmap::create_emit_map(cm).expect("emit map");
    let clen = emap.clen as usize;

    let mut end_e_ct = 0;
    for v in 0..=cm.m as usize {
        if v < cm.sttype.len() && cm.sttype[v] as i32 == E_ST {
            end_e_ct += 1;
        }
    }

    let mut cc2lins = vec![-1i32; clen + 1];
    let mut cc2rins = vec![-1i32; clen + 1];
    cc2lins[0] = 1; // ROOT_IL
    cc2rins[clen] = 2; // ROOT_IR
    for nd in 0..cm.nodes as usize {
        match cm.ndtype[nd] as i32 {
            x if x == MATP_ND => {
                cc2lins[emap.lpos[nd] as usize] = cm.nodemap[nd] + 4;
                cc2rins[(emap.rpos[nd] - 1) as usize] = cm.nodemap[nd] + 5;
            }
            x if x == MATL_ND => {
                cc2lins[emap.lpos[nd] as usize] = cm.nodemap[nd] + 2;
            }
            x if x == MATR_ND => {
                cc2rins[(emap.rpos[nd] - 1) as usize] = cm.nodemap[nd] + 2;
            }
            x if x == BEGR_ND => {
                cc2lins[emap.lpos[nd] as usize] = cm.nodemap[nd] + 1;
            }
            _ => {}
        }
    }

    let mut detach_ct = 0;
    for cc in 0..=clen {
        if cc2lins[cc] != -1 && cc2rins[cc] != -1 {
            detach_ct += 1;
            if do_check {
                if !cm_check_before_detaching(cm, cc2lins[cc], cc2rins[cc]) {
                    panic!("cm_check_before_detaching() returned false");
                }
            }
            if do_detach {
                if !cm_detach_state(cm, cc2lins[cc], cc2rins[cc]) {
                    panic!("cm_detach_state() returned false");
                }
            }
        }
    }
    detach_ct == end_e_ct
}

/// C: cm_detach_state()
fn cm_detach_state(cm: &mut CM, insert1: i32, insert2: i32) -> bool {
    if insert1 == insert2 {
        panic!("cm_detach_state insert1==insert2");
    }
    let to_detach;
    if cm.sttype[(insert1 + 1) as usize] as i32 == E_ST {
        to_detach = insert1;
    } else {
        if cm.sttype[(insert2 + 1) as usize] as i32 != E_ST {
            panic!("cm_detach_state: neither maps to END_E-1");
        }
        to_detach = insert2;
    }
    let x_offset = if cm.sttype[to_detach as usize] as i32 == IL_ST {
        0
    } else {
        if cm.stid[to_detach as usize] as i32 != MATP_IR {
            panic!("cm_detach_state: non-IL non-MATP_IR");
        }
        1
    };
    let mut y = cm.pnum[to_detach as usize] - 1;
    while y >= 1 {
        let xp = (cm.plast[to_detach as usize] - y) as usize;
        cm.t[xp][x_offset] = 0.0;
        let cnum = cm.cnum[xp] as usize;
        fnorm_slice(&mut cm.t[xp], cnum);
        y -= 1;
    }
    true
}

/// C: cm_check_before_detaching()
fn cm_check_before_detaching(cm: &CM, insert1: i32, insert2: i32) -> bool {
    if insert1 == insert2 {
        panic!("cm_check_before_detaching insert1==insert2");
    }
    let mut ret_val = false;
    let mut to_detach = -1;
    if cm.sttype[(insert1 + 1) as usize] as i32 == E_ST {
        ret_val = true;
        to_detach = insert1;
    }
    if cm.sttype[(insert2 + 1) as usize] as i32 == E_ST {
        if ret_val {
            panic!("cm_check_before_detaching: both map to END_E-1");
        }
        ret_val = true;
        to_detach = insert2;
    }
    if ret_val {
        for i in 0..K {
            if cm.e[to_detach as usize][i].abs() > 0.000001 {
                panic!("to_detach e[{}] nonzero", i);
            }
        }
        for yoffset in 0..cm.cnum[to_detach as usize] as usize {
            if cm.t[to_detach as usize][yoffset].abs() > 0.000001 {
                panic!("to_detach t[{}] nonzero", yoffset);
            }
        }
    }
    ret_val
}

fn fnorm_slice(v: &mut [f32], n: usize) {
    let sum: f32 = v[..n].iter().sum();
    if sum != 0.0 {
        for x in &mut v[..n] {
            *x /= sum;
        }
    } else {
        for x in &mut v[..n] {
            *x = 1.0 / n as f32;
        }
    }
}

/// C: cmbuild.c:flatten_insert_emissions(). Set IL/IR emission probs to null.
pub fn flatten_insert_emissions(cm: &mut CM) {
    // esl_vec_FNorm(cm->null, K)
    let sum: f32 = cm.null.iter().sum();
    if sum != 0.0 {
        for x in &mut cm.null {
            *x /= sum;
        }
    }
    let null = cm.null;
    for v in 0..cm.m as usize {
        let st = cm.sttype[v] as i32;
        if st == IL_ST || st == IR_ST {
            for x in &mut cm.e[v] {
                *x = 0.0;
            }
            for a in 0..K {
                cm.e[v][a] = null[a];
            }
        }
    }
}

// =============================================================================
// configure_model glue (cm_modelconfig.c:cm_ConfigureSub, QDB/W path only)
// =============================================================================

use crate::cm_qdband::{calculate_query_dependent_bands, QdbInfo};
use crate::constants::IMPOSSIBLE;

/// C: cm_ConfigureSub() restricted to the non-local cmbuild path: compute QDBs
/// (both beta sets) and W (from cm.w_beta), enforce el_selfsc*W >= IMPOSSIBLE,
/// then logoddsify. The p7/CP9 filter build is deferred (not needed for the CM
/// body; see bin/cmbuild.rs).
pub fn configure_qdb_and_w(cm: &mut CM) {
    let mut qi = QdbInfo::new(cm.m as usize, cm.clen);
    qi.beta1 = cm.qdb_beta1;
    qi.beta2 = cm.qdb_beta2;
    let res = calculate_query_dependent_bands(cm, Some(qi), cm.w_beta, false, false)
        .expect("QDB band calculation failed");
    cm.w = res.w;
    if let Some(qi) = res.qdbinfo {
        cm.dmin1 = qi.dmin1;
        cm.dmax1 = qi.dmax1;
        cm.dmin2 = qi.dmin2;
        cm.dmax2 = qi.dmax2;
    }
    // C: if (el_selfsc * W) < IMPOSSIBLE then el_selfsc = IMPOSSIBLE/(W+1)
    if (cm.el_selfsc as f64) * (cm.w as f64) < IMPOSSIBLE {
        cm.el_selfsc = (IMPOSSIBLE / (cm.w as f64 + 1.0)) as f32;
    }
    cm.flags |= crate::cm::CM_W;
    cm.cm_logoddsify();
}
