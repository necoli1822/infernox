//! Faithful port of Infernal's CM rebalancing (original/src/cm.c:CMRebalance).
//!
//! `CMRebalance()` takes a freshly-built, count-based CM (structure + null model +
//! map/rf/consensus set; transition/emission COUNTS in cm.t/cm.e — which are all
//! zero at the point cmbuild::build_model calls this, right after CMSetNullModel and
//! before Transmogrify/counting) and returns a NEW CM with the *same* states and
//! nodes, but with the state indices RENUMBERED so that at every bifurcation the
//! lighter (smaller/less deck-hungry) subtree becomes the BEGL (left) child that is
//! visited first. This improves divide-and-conquer (D&C) alignment balance.
//!
//! The renumbering is purely a state reindexing: nodes keep their preorder numbering
//! (the guide tree is unchanged, cm.c:2008-2015). All probabilities/counts,
//! emissions, transitions, the null model, and the map/rf/consensus annotation are
//! preserved verbatim — only the per-state arrays move under the `newidx[]` map.
//!
//! C source: original/src/cm.c
//!   - CMRebalance()            cm.c:1840-2031  (main routine, ported below)
//!   - CMSubtreeFindEnd()       cm.c:1031-1042  (helper, ported below)
//!   - CMSubtreeCountStatetype() cm.c:1012-1025 (helper, ported below)
//!
//! Balancing rule (cm.c:1878-1890): a "weight" wgt[v] = number of extra CYK decks
//! required to compute the subgraph rooted at v, computed bottom-up:
//!   - E state:  wgt = 1                       (unbifurcated segment base case)
//!   - B state:  wgt = 1 + min(wgt[left], wgt[right])
//!   - else:     wgt = wgt[v+1]                (propagate up to the enclosing S)
//! At each bifurcation the child with the SMALLER weight is visited first. The
//! tie-break is `<=` on the LEFT child (cm.c:1955): if wgt[left] <= wgt[right] the
//! left child is visited first (i.e. ties keep the left child on the left).

use crate::cm::{CM, CM_MAXCONNECT, PAIR_EMIT_SIZE};
use crate::constants::{B_ST, E_ST};

/// Faithful port of `CMRebalance()` (cm.c:1840).
///
/// Returns a new CM identical in M / nodes to `cm`, with states renumbered so the
/// lighter subtree is the BEGL (left) child at every bifurcation.
///
/// The C prototype is `int CMRebalance(CM_t *cm, char *errbuf, CM_t **ret_new_cm)`,
/// returning eslOK and passing the new CM out through `ret_new_cm`. The initial
/// `cm_nonconfigured_Verify()` guard (cm.c:1851) only rejects an already-configured
/// CM; at cmbuild's build_model call site the CM is freshly built and always passes,
/// so we drop the errbuf/Result plumbing and return the new CM directly.
pub fn cm_rebalance(cm: &CM) -> CM {
    let m = cm.m as usize;
    let nodes = cm.nodes as usize;

    // -------------------------------------------------------------------------
    // Create the new model. Copy information that's unchanged by renumbering the
    // CM. (cm.c:1853-1876)
    //
    // C: new = CreateCM(cm->nodes, cm->M, cm->clen, cm->abc);
    // -------------------------------------------------------------------------
    let mut new = CM::new(cm.m, cm.nodes);

    // cm.c:1857-1865 — identifiers and annotation (unchanged by renumbering).
    new.name = cm.name.clone();
    new.acc = cm.acc.clone();
    new.desc = cm.desc.clone();
    new.rf = cm.rf.clone(); // C: esl_strdup(cm->rf, ...)
    new.consensus = cm.consensus.clone(); // C: esl_strdup(cm->consensus, ...)
    // C: if(cm->map != NULL) { ESL_ALLOC(...); esl_vec_ICopy(cm->map, cm->clen+1, ...); }
    if !cm.map.is_empty() {
        new.map = cm.map.clone();
    }
    // Writer-only metadata carried verbatim (no direct C analogue in this loop but
    // "unchanged by renumbering"): DATE/COM/checksum lines.
    new.ctime = cm.ctime.clone();
    new.comlog = cm.comlog.clone();
    new.checksum = cm.checksum;

    // cm.c:1867-1874 — flags & scalar training stats.
    new.flags = cm.flags; // preserves CM_MAP / CM_RF / CM_CONS etc.
    new.clen = cm.clen;
    new.w = cm.w;
    new.nseq = cm.nseq;
    new.eff_nseq = cm.eff_nseq;
    // C gates ga/tc/nc on CMH_GA / CMH_TC / CMH_NC. They default to 0 in a fresh CM
    // and are copied verbatim regardless; the flags in `new.flags` already record
    // whether they are meaningful.
    new.ga = cm.ga;
    new.tc = cm.tc;
    new.nc = cm.nc;

    // cm.c:1876 — null model: for (x=0; x<cm->abc->K; x++) new->null[x] = cm->null[x];
    new.null = cm.null;

    // Additional per-model scalars that are unaffected by the state renumbering.
    // (Not in the C loop because the C struct stores them elsewhere, but they must
    // survive the copy so downstream cmbuild stages see the same model.)
    new.el_selfsc = cm.el_selfsc;
    new.pbegin = cm.pbegin;
    new.pend = cm.pend;
    new.w_beta = cm.w_beta;
    new.qdb_beta1 = cm.qdb_beta1;
    new.qdb_beta2 = cm.qdb_beta2;
    new.n2_omega = cm.n2_omega;
    new.n3_omega = cm.n3_omega;
    new.efp7gf_tau = cm.efp7gf_tau;
    new.efp7gf_lambda = cm.efp7gf_lambda;

    // -------------------------------------------------------------------------
    // Calculate "weights" (# of required extra decks) on every B and S state.
    // Recursive rule: 1 + min(wgt[left], wgt[right]).  (cm.c:1878-1890)
    // -------------------------------------------------------------------------
    let mut wgt = vec![0i32; m]; // C: ESL_ALLOC(wgt, sizeof(int) * cm->M);
    for v in (0..m).rev() {
        if cm.sttype[v] as i32 == E_ST {
            // initialize unbifurcated segments with 1
            wgt[v] = 1;
        } else if cm.sttype[v] as i32 == B_ST {
            // "cfirst"=left S child. "cnum"=right S child.
            let l = wgt[cm.cfirst[v] as usize];
            let r = wgt[cm.cnum[v] as usize];
            wgt[v] = 1 + if l < r { l } else { r }; // ESL_MIN
        } else {
            // all other states propagate up to S
            wgt[v] = wgt[v + 1];
        }
    }

    // -------------------------------------------------------------------------
    // Preorder-traverse the new CM. At each bifurcation visit the S with minimum
    // weight first. `v` indexes the OLD CM, hopping around via this traversal order
    // and a pushdown stack; `nv` indexes the NEW CM, moving 0..M-1 in preorder.
    // (cm.c:1892-1986)
    // -------------------------------------------------------------------------
    let mut v: i32 = 0;
    let mut z: i32 = cm.m - 1;
    // C: pda = esl_stack_ICreate();  — a LIFO int stack. Vec<i32> push/pop = LIFO.
    let mut pda: Vec<i32> = Vec::new();
    let mut newidx = vec![0i32; m]; // newidx[v] = old state v's index in the new CM

    for nv in 0..(m as i32) {
        let vu = v as usize;
        let nvu = nv as usize;

        // Keep a map of where old states go in the new CM: old state v -> newidx[v].
        // Guaranteed one-to-one. (cm.c:1905-1909)
        newidx[vu] = nv;

        // Copy old v to new nv — first the easy stuff, unaffected by renumbering.
        // (cm.c:1911-1925)
        new.sttype[nvu] = cm.sttype[vu];
        new.ndidx[nvu] = cm.ndidx[vu];
        new.stid[nvu] = cm.stid[vu];
        new.pnum[nvu] = cm.pnum[vu];
        // C: for (x=0; x<MAXCONNECT; x++) { new->t[nv][x]=cm->t[v][x]; new->tsc[nv][x]=cm->t[v][x]; }
        // Note the C quirk: tsc is seeded from the COUNTS in t (log-odds are filled in
        // later by cm_Logoddsify). At this build stage t is all zeros, but we mirror
        // the exact C assignment for faithfulness.
        for x in 0..CM_MAXCONNECT {
            new.t[nvu][x] = cm.t[vu][x];
            new.tsc[nvu][x] = cm.t[vu][x];
        }
        // C: for (x=0; x<cm->abc->K*cm->abc->K; x++) { new->e[nv][x]=cm->e[v][x]; new->esc[nv][x]=cm->e[v][x]; }
        // K*K = 4*4 = 16 = PAIR_EMIT_SIZE (row length for both single and pair states).
        for x in 0..PAIR_EMIT_SIZE {
            new.e[nvu][x] = cm.e[vu][x];
            new.esc[nvu][x] = cm.e[vu][x];
        }

        // The plast connection for nv, to the last of 1-6 parent states, via newidx.
        // (cm.c:1927-1931)
        if nv != 0 {
            new.plast[nvu] = newidx[cm.plast[vu] as usize];
        } else {
            new.plast[nvu] = -1; // ROOT
        }

        // Figure out next v, and make cfirst/cnum connections. (cm.c:1933-1985)
        if cm.sttype[vu] as i32 == B_ST {
            // Remember the CM overload: cfirst = idx of left child; cnum = idx of
            // right child. If we visit left w first, cfirst=nv+1; if we visit right y
            // first, cnum=nv+1. The # of states in the first subgraph we visit is
            // y-w, so the second child index is nv+y-w+1. (cm.c:1936-1948)
            let w = cm.cfirst[vu]; // left child of v
            let y = cm.cnum[vu]; // right child of v

            if wgt[w as usize] <= wgt[y as usize] {
                // left (w) lighter or same weight? visit w first, defer y. (cm.c:1955-1963)
                pda.push(y);
                pda.push(z);
                v = w;
                z = y - 1;
                new.cfirst[nvu] = nv + 1; // left child is nv+1
                new.cnum[nvu] = nv + y - w + 1;
            } else {
                // right (y) lighter? visit y first, defer w. (cm.c:1964-1971)
                pda.push(w);
                pda.push(y - 1);
                v = y; // z unchanged
                new.cfirst[nvu] = nv + z - y + 2;
                new.cnum[nvu] = nv + 1; // right child is nv+1
            }
        } else if cm.sttype[vu] as i32 == E_ST {
            // No children; pop the next (v, z) off the stack. (cm.c:1973-1979)
            new.cfirst[nvu] = -1;
            new.cnum[nvu] = 0;
            // C pops z first, then v (esl_stack_IPop is LIFO; z was pushed last).
            // On the FINAL E state the stack is empty (#E = #B + 1); C's
            // esl_stack_IPop returns eslEOD and leaves v/z unchanged, then the
            // for loop exits. Mirror that: keep current values on underflow.
            z = pda.pop().unwrap_or(z);
            v = pda.pop().unwrap_or(v);
        } else {
            // Ordinary state: next v is v+1. cfirst via the offset in the old model;
            // cnum is unchanged. (cm.c:1980-1985)
            new.cfirst[nvu] = nv + (cm.cfirst[vu] - v); // nv + (cfirst[v] - v)
            new.cnum[nvu] = cm.cnum[vu]; // unchanged
            v += 1;
        }
    }

    // -------------------------------------------------------------------------
    // Renumbered begin/end transition distributions, via newidx[v]. (cm.c:1988-1999)
    // At this build stage these are the CM::new defaults, but we remap faithfully.
    // -------------------------------------------------------------------------
    for v in 0..m {
        let nvu = newidx[v] as usize;
        new.begin[nvu] = cm.begin[v];
        new.beginsc[nvu] = cm.beginsc[v];
        new.end[nvu] = cm.end[v];
        new.endsc[nvu] = cm.endsc[v];
    }
    // C also remaps ibeginsc/iendsc; those integer-score arrays are empty (unbuilt)
    // at this stage in the Rust CM, so only remap when present.
    if cm.ibeginsc.len() == m {
        new.ibeginsc = vec![0i32; m];
        for v in 0..m {
            new.ibeginsc[newidx[v] as usize] = cm.ibeginsc[v];
        }
    }
    if cm.iendsc.len() == m {
        new.iendsc = vec![0i32; m];
        for v in 0..m {
            new.iendsc[newidx[v] as usize] = cm.iendsc[v];
        }
    }

    // QDB bands, remapped via newidx (cm.c:2000-2006, the qdbinfo dmin/dmax loops).
    for v in 0..m {
        let nvu = newidx[v] as usize;
        new.dmin1[nvu] = cm.dmin1[v];
        new.dmax1[nvu] = cm.dmax1[v];
        new.dmin2[nvu] = cm.dmin2[v];
        new.dmax2[nvu] = cm.dmax2[v];
    }

    // -------------------------------------------------------------------------
    // Guide tree numbering is unchanged (still preorder). Associate nodes with the
    // new state numbering. (cm.c:2008-2015)
    // -------------------------------------------------------------------------
    for x in 0..nodes {
        new.nodemap[x] = newidx[cm.nodemap[x] as usize];
        new.ndtype[x] = cm.ndtype[x];
    }

    // lchild/rchild are this codebase's explicit copies of the CM overload
    // (cfirst=left S child, cnum=right S child) for B states; the CM reader sets
    // lchild=cfirst, rchild=cnum. Mirror that under the new numbering so the two
    // representations stay consistent. (No direct C analogue — the C struct only has
    // the cfirst/cnum overload.)
    for nv in 0..m {
        if new.sttype[nv] as i32 == B_ST {
            new.lchild[nv] = new.cfirst[nv];
            new.rchild[nv] = new.cnum[nv];
        }
    }

    new
}

/// Faithful port of `CMSubtreeFindEnd()` (cm.c:1031-1042).
///
/// Walks forward from state `r` (inclusive), counting bifurcations (B) as extra
/// starts and ends (E) as satisfied starts, and returns the index of the E state
/// that closes the subtree rooted at `r`.
pub fn cm_subtree_find_end(cm: &CM, r: i32) -> i32 {
    let mut unsatisfied_starts = 1i32;
    let mut r = r;
    while unsatisfied_starts != 0 {
        if cm.sttype[r as usize] as i32 == B_ST {
            unsatisfied_starts += 1;
        }
        if cm.sttype[r as usize] as i32 == E_ST {
            unsatisfied_starts -= 1;
        }
        r += 1;
    }
    r - 1
}

/// Faithful port of `CMSubtreeCountStatetype()` (cm.c:1012-1025).
///
/// Counts how many states of type `stype` occur in the subtree rooted at state `v`
/// (inclusive), using the same B/E bracket-matching walk as `cm_subtree_find_end`.
pub fn cm_subtree_count_statetype(cm: &CM, v: i32, stype: i32) -> i32 {
    let mut unsatisfied_starts = 1i32;
    let mut count = 0i32;
    let mut v = v;
    while unsatisfied_starts != 0 {
        let st = cm.sttype[v as usize] as i32;
        if st == B_ST {
            unsatisfied_starts += 1;
        }
        if st == E_ST {
            unsatisfied_starts -= 1;
        }
        if st == stype {
            count += 1;
        }
        v += 1;
    }
    count
}
