//! Consensus-sequence construction for a covariance model.
//!
//! Faithful byte-parity port of Infernal 1.1.5's consensus display machinery:
//!   - `display.c:CreateCMConsensus()`            (l.653)
//!   - `display.c:createMultifurcationOrderChart()` (l.882)
//!   - `display.c:createFaceCharts()`             (l.953)
//!   - `cm.c:cm_SetConsensus()`                   (l.2733)
//!
//! Produces the display consensus sequence `cseq` (length = cm->clen) that is
//! used to populate `cm.consensus`. We build the full `CMConsensus_t` struct
//! faithfully even though downstream we only consume `cseq`.
//!
//! PDA move-type constants (infernal.h l.254-256):
//!   #define PDA_RESIDUE 0
//!   #define PDA_STATE   1
//!   #define PDA_MARKER  2

use crate::cm::{CM, CM_CONS};
use crate::constants::{
    B_ST, E_ST,
    BEGL_ND, BEGR_ND, BIF_ND, END_ND, MATP_ND, ROOT_ND,
    BIF_B, END_E, MATP_MP, MATL_ML, MATR_MR,
};

/* infernal.h:254-256 */
const PDA_RESIDUE: i32 = 0;
const PDA_STATE: i32 = 1;
const PDA_MARKER: i32 = 2;

/// abc->sym[x] for the RNA alphabet, x in 0..K (K = 4).
/// (Easel: eslRNA sym[0..3] = 'A','C','G','U'.)
const SYM: [u8; 4] = [b'A', b'C', b'G', b'U'];

/// abc->K, the canonical alphabet size.
const K: usize = 4;

/// Structure: CMConsensus_t (infernal.h l.370-377)
///
/// ```c
/// typedef struct consensus_s {
///   char *cseq;   /* consensus sequence display string; 0..clen-1  */
///   char *cstr;   /* consensus structure display string; 0..clen-1 */
///   int  *ct;     /* Zuker-style ct pairing map; [0..clen-1]       */
///   int  *lpos;   /* maps node->consensus position; 0..nodes-1     */
///   int  *rpos;   /* maps node->consensus position; 0..nodes-1     */
///   int   clen;   /* length of cseq, cstr                          */
/// } CMConsensus_t;
/// ```
#[derive(Debug, Clone)]
pub struct CMConsensus {
    /// consensus sequence display string; 0..clen-1
    pub cseq: Vec<u8>,
    /// consensus structure display string; 0..clen-1
    pub cstr: Vec<u8>,
    /// Zuker-style ct pairing map; [0..clen-1], -1 = no partner
    pub ct: Vec<i32>,
    /// maps node->left consensus position; 0..nodes-1
    pub lpos: Vec<i32>,
    /// maps node->right consensus position; 0..nodes-1
    pub rpos: Vec<i32>,
    /// length of cseq, cstr
    pub clen: i32,
}

/// esl_vec_FArgMax(v, n): index of the max element among v[0..n).
/// C scans 0..n keeping strict `>`, so the *first* maximum wins on ties.
///
/// ```c
/// int esl_vec_FArgMax(const float *vec, int n) {
///   int i, best = 0;
///   for (i = 1; i < n; i++) if (vec[i] > vec[best]) best = i;
///   return best;
/// }
/// ```
fn esl_vec_f_arg_max(v: &[f32], n: usize) -> usize {
    let mut best = 0usize;
    let mut i = 1usize;
    while i < n {
        if v[i] > v[best] {
            best = i;
        }
        i += 1;
    }
    best
}

/// display.c:createMultifurcationOrderChart() (l.882)
///
/// Calculates the degree of multifurcation ("height") beneath the master
/// subtree rooted at every node. Terminal stems have value 0; a stem above a
/// multifurcation into all terminal stems has value 1; and so on. Walks nodes
/// from last to first, tracking whether a segment contains pairs, and at each
/// BIF takes the max of the two child subtree heights (each plus whether that
/// child segment has pairs). Returns a [0..nodes-1] array of orders.
///
/// (The original C header notes "THIS FUNCTION IS BUGGY (Sat Jun 1 2002)".)
///
/// ```c
/// static int *
/// createMultifurcationOrderChart(CM_t *cm)
/// {
///   int status;
///   int  v, nd, left, right;
///   int *height;
///   int *seg_has_pairs;
///
///   ESL_ALLOC(height,        sizeof(int) * cm->nodes);
///   ESL_ALLOC(seg_has_pairs, sizeof(int) * cm->nodes);
///   for (nd = cm->nodes-1; nd >= 0; nd--)
///     {
///       v = cm->nodemap[nd];
///
///       if       (cm->stid[v] == MATP_MP) seg_has_pairs[nd] = TRUE;
///       else if  (cm->stid[v] == END_E)   seg_has_pairs[nd] = FALSE;
///       else if  (cm->stid[v] == BIF_B)   seg_has_pairs[nd] = FALSE;
///       else                              seg_has_pairs[nd] = seg_has_pairs[nd+1];
///
///       if (cm->stid[v] == END_E)
///         height[nd]        = 0;
///       else if (cm->stid[v] == BIF_B)
///         {
///           left  = cm->ndidx[cm->cfirst[v]];
///           right = cm->ndidx[cm->cnum[v]];
///           height[nd] = ESL_MAX(height[left] + seg_has_pairs[left],
///                                height[right] + seg_has_pairs[right]);
///         }
///       else
///         height[nd] = height[nd+1];
///     }
///   free(seg_has_pairs);
///   return height;
/// }
/// ```
fn create_multifurcation_order_chart(cm: &CM) -> Vec<i32> {
    let nodes = cm.nodes as usize;
    let mut height: Vec<i32> = vec![0; nodes];
    let mut seg_has_pairs: Vec<i32> = vec![0; nodes];

    let mut nd = cm.nodes - 1;
    while nd >= 0 {
        let ndu = nd as usize;
        let v = cm.nodemap[ndu] as usize;
        let stid = cm.stid[v] as i32;

        if stid == MATP_MP {
            seg_has_pairs[ndu] = 1; /* TRUE */
        } else if stid == END_E {
            seg_has_pairs[ndu] = 0; /* FALSE */
        } else if stid == BIF_B {
            seg_has_pairs[ndu] = 0; /* FALSE */
        } else {
            seg_has_pairs[ndu] = seg_has_pairs[ndu + 1];
        }

        if stid == END_E {
            height[ndu] = 0;
        } else if stid == BIF_B {
            let left = cm.ndidx[cm.cfirst[v] as usize] as usize;
            let right = cm.ndidx[cm.cnum[v] as usize] as usize;
            let a = height[left] + seg_has_pairs[left];
            let b = height[right] + seg_has_pairs[right];
            height[ndu] = if a >= b { a } else { b }; /* ESL_MAX */
        } else {
            height[ndu] = height[ndu + 1];
        }

        nd -= 1;
    }
    height
}

/// display.c:createFaceCharts() (l.953)
///
/// Computes `inface` and `outface` counts for each node. `inface` (bottom-up
/// pass) is the number of structural faces strictly below a node in its
/// descendant subtrees (0 => external/closing-bp/hairpin-loop). `outface`
/// (top-down pass) is the number of faces above a node excluding its own
/// subtree (0 => external). Together they classify each node's structural
/// context (external ss, hairpin loop, bulge/interior, multiloop).
///
/// NOTE (faithful bug reproduction): the original C BEGR_nd branch reads a
/// stale `y` that is only ever assigned inside the BEGL_nd branch. C declares
/// `int v, y;` uninitialized at function scope, so the value of `y` seen in a
/// BEGR_nd iteration is whatever was last written by a preceding BEGL_nd
/// iteration (or indeterminate if none preceded). We reproduce this exactly by
/// carrying `y` across loop iterations, initialized to 0. This only affects
/// `outface` (hence structure annotation `cstr`), never `cseq`.
///
/// ```c
/// static void
/// createFaceCharts(CM_t *cm, int **ret_inface, int **ret_outface)
/// {
///   int  status;
///   int *inface;
///   int *outface;
///   int  nd, left, right, parent;
///   int  v,y;
///
///   ESL_ALLOC(inface,  sizeof(int) * cm->nodes);
///   ESL_ALLOC(outface, sizeof(int) * cm->nodes);
///
///   for (nd = cm->nodes-1; nd >= 0; nd--)
///     {
///       v = cm->nodemap[nd];
///       if      (cm->ndtype[nd] == END_nd) inface[nd] = 0;
///       else if (cm->ndtype[nd] == BIF_nd) {
///         left  = cm->ndidx[cm->cfirst[v]];
///         right = cm->ndidx[cm->cnum[v]];
///         inface[nd] = inface[left] + inface[right];
///       } else {
///         if (cm->ndtype[nd+1] == MATP_nd) inface[nd] = 1;
///         else                             inface[nd] = inface[nd+1];
///       }
///     }
///
///   for (nd = 0; nd < cm->nodes; nd++)
///     {
///       v = cm->nodemap[nd];
///       if      (cm->ndtype[nd] == ROOT_nd) outface[nd] = 0;
///       else if (cm->ndtype[nd] == BEGL_nd)
///         {
///           parent = cm->ndidx[cm->plast[v]];
///           y      = cm->nodemap[parent];
///           right  = cm->ndidx[cm->cnum[y]];
///           outface[nd] = outface[parent] + inface[right];
///         }
///       else if (cm->ndtype[nd] == BEGR_nd)
///         {
///           parent = cm->ndidx[cm->plast[v]];
///           left   = cm->ndidx[cm->cfirst[y]];
///           outface[nd] = outface[parent] + inface[left];
///         }
///       else
///         {
///           parent = nd-1;
///           if (cm->ndtype[parent] == MATP_nd) outface[nd] = 1;
///           else                               outface[nd] = outface[parent];
///         }
///     }
///
///   *ret_inface  = inface;
///   *ret_outface = outface;
///   return;
/// }
/// ```
fn create_face_charts(cm: &CM) -> (Vec<i32>, Vec<i32>) {
    let nodes = cm.nodes as usize;
    let mut inface: Vec<i32> = vec![0; nodes];
    let mut outface: Vec<i32> = vec![0; nodes];

    /* inface: bottom-up. */
    let mut nd = cm.nodes - 1;
    while nd >= 0 {
        let ndu = nd as usize;
        let v = cm.nodemap[ndu] as usize;
        let ndtype = cm.ndtype[ndu] as i32;
        if ndtype == END_ND {
            inface[ndu] = 0;
        } else if ndtype == BIF_ND {
            let left = cm.ndidx[cm.cfirst[v] as usize] as usize;
            let right = cm.ndidx[cm.cnum[v] as usize] as usize;
            inface[ndu] = inface[left] + inface[right];
        } else {
            if cm.ndtype[ndu + 1] as i32 == MATP_ND {
                inface[ndu] = 1;
            } else {
                inface[ndu] = inface[ndu + 1];
            }
        }
        nd -= 1;
    }

    /* outface: top-down. `y` carried across iterations (see NOTE above). */
    let mut y: i32 = 0;
    for ndu in 0..nodes {
        let v = cm.nodemap[ndu] as usize;
        let ndtype = cm.ndtype[ndu] as i32;
        if ndtype == ROOT_ND {
            outface[ndu] = 0;
        } else if ndtype == BEGL_ND {
            let parent = cm.ndidx[cm.plast[v] as usize] as usize;
            y = cm.nodemap[parent];
            let right = cm.ndidx[cm.cnum[y as usize] as usize] as usize;
            outface[ndu] = outface[parent] + inface[right];
        } else if ndtype == BEGR_ND {
            let parent = cm.ndidx[cm.plast[v] as usize] as usize;
            /* faithful: uses stale `y` from a prior BEGL iteration */
            let left = cm.ndidx[cm.cfirst[y as usize] as usize] as usize;
            outface[ndu] = outface[parent] + inface[left];
        } else {
            let parent = ndu - 1;
            if cm.ndtype[parent] as i32 == MATP_ND {
                outface[ndu] = 1;
            } else {
                outface[ndu] = outface[parent];
            }
        }
    }

    (inface, outface)
}

/// display.c:CreateCMConsensus() (l.653)
///
/// Builds displayable consensus sequence (`cseq`) and structure (`cstr`)
/// strings plus node->consensus-position maps, via a PDA (pushdown automaton)
/// traversal of the model's master (consensus) subtree. Consensus residues are
/// the maximum-scoring emission(s) per node (`cm->esc[v]` via
/// `esl_vec_FArgMax`); residues weaker than the thresholds are lowercased
/// (pairs < pthresh=3.0, singlets < sthresh=1.0). Structure characters are
/// chosen from the multifurcation-order chart (pairs) and inside/outside face
/// charts (singlets).
///
/// Returns the full CMConsensus_t. Returns None on contract failure (CM lacks
/// log-odds scores), matching the C which returns NULL.
///
/// ```c
/// CMConsensus_t *
/// CreateCMConsensus(CM_t *cm, const ESL_ALPHABET *abc)
/// { ... }   /* see below, transcribed inline */
/// ```
pub fn create_cm_consensus_full(cm: &CM) -> Option<CMConsensus> {
    /* thresholds, hard-coded (display.c l.680-681). */
    let pthresh: f32 = 3.0;
    let sthresh: f32 = 1.0;

    /* Contract check. CM must have log odds scores.
     * (C also checks abc compatibility; here abc == cm->abc RNA, K==K, so OK.)
     *   if(! (cm->flags & CMH_BITS)) return NULL;
     * We express the equivalent guard: without bit scores esc is meaningless.
     * Rust: CM_BITS flag is checked by the caller path; we mirror the NULL
     * return contract by requiring esc to be present.
     */
    if cm.esc.is_empty() {
        return None;
    }

    let nodes = cm.nodes as usize;

    /* lpos/rpos: maps node -> consensus position, init to -1.
     *   for (nd = 0; nd < cm->nodes; nd++) lpos[nd] = rpos[nd] = -1;
     */
    let mut lpos: Vec<i32> = vec![-1; nodes];
    let mut rpos: Vec<i32> = vec![-1; nodes];

    /* cseq/cstr/ct grow append-only in lockstep with cpos; ct is also
     * back-patched at earlier indices. In C these are realloc'd by +100;
     * we use Vec push instead (functionally identical). */
    let mut cseq: Vec<u8> = Vec::new();
    let mut cstr: Vec<u8> = Vec::new();
    let mut ct: Vec<i32> = Vec::new();
    let mut cpos: i32 = 0;

    let multiorder = create_multifurcation_order_chart(cm);
    let (inface, outface) = create_face_charts(cm);

    /* PDA. ESL_STACK of ints, LIFO. Modeled as Vec<i32> with push()/pop().
     *   if ((status = esl_stack_IPush(pda, 0)) != eslOK) goto ERROR;
     *   if ((status = esl_stack_IPush(pda, PDA_STATE)) != eslOK) goto ERROR;
     */
    let mut pda: Vec<i32> = Vec::new();
    pda.push(0);
    pda.push(PDA_STATE);

    /* while (esl_stack_IPop(pda, &type) != eslEOD) */
    while let Some(ptype) = pda.pop() {
        if ptype == PDA_RESIDUE {
            /* esl_stack_IPop(pda, &x); rchar  = (char) x;
             * esl_stack_IPop(pda, &x); rstruc = (char) x;
             * esl_stack_IPop(pda, &pairpartner);
             * esl_stack_IPop(pda, &nd);
             */
            let rchar = pda.pop().unwrap() as u8;
            let rstruc = pda.pop().unwrap() as u8;
            let pairpartner = pda.pop().unwrap();
            let nd = pda.pop().unwrap();

            rpos[nd as usize] = cpos;
            /* cseq[cpos] = rchar; cstr[cpos] = rstruc; ct[cpos] = pairpartner; */
            cseq.push(rchar);
            cstr.push(rstruc);
            ct.push(pairpartner);
            /* if (pairpartner != -1) ct[pairpartner] = cpos; */
            if pairpartner != -1 {
                ct[pairpartner as usize] = cpos;
            }
            cpos += 1;
        } else if ptype == PDA_MARKER {
            /* esl_stack_IPop(pda, &nd); rpos[nd] = cpos-1; */
            let nd = pda.pop().unwrap();
            rpos[nd as usize] = cpos - 1;
        } else if ptype == PDA_STATE {
            /* esl_stack_IPop(pda, &v); nd = cm->ndidx[v]; */
            let mut v = pda.pop().unwrap();
            let nd = cm.ndidx[v as usize];

            /* lchar = rchar = lstruc = rstruc = 0; (0 == "no emission") */
            let mut lchar: u8 = 0;
            let mut rchar: u8 = 0;
            let mut lstruc: u8 = 0;
            let mut rstruc: u8 = 0;

            let stid = cm.stid[v as usize] as i32;
            let ndu = nd as usize;

            /* Determine what we emit: MATP, MATL, MATR consensus states only. */
            if stid == MATP_MP {
                /* x = esl_vec_FArgMax(cm->esc[v], abc->K*abc->K);
                 * lchar = abc->sym[x / abc->K];
                 * rchar = abc->sym[x % abc->K];
                 * if (cm->esc[v][x] < pthresh) { lchar/rchar = tolower(...); }
                 */
                let x = esl_vec_f_arg_max(&cm.esc[v as usize], K * K);
                lchar = SYM[x / K];
                rchar = SYM[x % K];
                if cm.esc[v as usize][x] < pthresh {
                    lchar = lchar.to_ascii_lowercase();
                    rchar = rchar.to_ascii_lowercase();
                }
                /* switch (multiorder[nd]) { ... } */
                match multiorder[ndu] {
                    0 => {
                        lstruc = b'<';
                        rstruc = b'>';
                    }
                    1 => {
                        lstruc = b'(';
                        rstruc = b')';
                    }
                    2 => {
                        lstruc = b'[';
                        rstruc = b']';
                    }
                    _ => {
                        lstruc = b'{';
                        rstruc = b'}';
                    }
                }
            } else if stid == MATL_ML {
                /* x = esl_vec_FArgMax(cm->esc[v], cm->abc->K);
                 * lchar = abc->sym[x];
                 * if (cm->esc[v][x] < sthresh) lchar = tolower(...);
                 */
                let x = esl_vec_f_arg_max(&cm.esc[v as usize], K);
                lchar = SYM[x];
                if cm.esc[v as usize][x] < sthresh {
                    lchar = lchar.to_ascii_lowercase();
                }
                if outface[ndu] == 0 {
                    lstruc = b':'; /* external ss */
                } else if inface[ndu] == 0 && outface[ndu] == 1 {
                    lstruc = b'_'; /* hairpin loop */
                } else if inface[ndu] == 1 && outface[ndu] == 1 {
                    lstruc = b'-'; /* bulge/interior */
                } else {
                    lstruc = b','; /* multiloop */
                }
                rstruc = b' ';
            } else if stid == MATR_MR {
                /* x = esl_vec_FArgMax(cm->esc[v], cm->abc->K);
                 * rchar = abc->sym[x];
                 * if (cm->esc[v][x] < sthresh) rchar = tolower(...);
                 */
                let x = esl_vec_f_arg_max(&cm.esc[v as usize], K);
                rchar = SYM[x];
                if cm.esc[v as usize][x] < sthresh {
                    rchar = rchar.to_ascii_lowercase();
                }
                if outface[ndu] == 0 {
                    rstruc = b':'; /* external ss */
                } else if inface[ndu] == 0 && outface[ndu] == 1 {
                    rstruc = b'?'; /* doesn't happen */
                } else if inface[ndu] == 1 && outface[ndu] == 1 {
                    rstruc = b'-'; /* bulge/interior */
                } else {
                    rstruc = b','; /* multiloop */
                }
                lstruc = b' ';
            }

            /* Emit. A left base now; a right base deferred onto PDA.
             * lpos[nd] = cpos;  (always set, even for nonemitters)
             */
            lpos[ndu] = cpos;
            if lchar != 0 {
                /* cseq[cpos] = lchar; cstr[cpos] = lstruc; ct[cpos] = -1; cpos++; */
                cseq.push(lchar);
                cstr.push(lstruc);
                ct.push(-1); /* will be overwritten if right guy processed */
                cpos += 1;
            }
            if rchar != 0 {
                /* push (nd, pairpartner, rstruc, rchar, PDA_RESIDUE) */
                pda.push(nd);
                if lchar != 0 {
                    pda.push(cpos - 1);
                } else {
                    pda.push(-1);
                }
                pda.push(rstruc as i32);
                pda.push(rchar as i32);
                pda.push(PDA_RESIDUE);
            } else {
                /* push (nd, PDA_MARKER) */
                pda.push(nd);
                pda.push(PDA_MARKER);
            }

            /* Transit - to consensus states only. */
            if cm.sttype[v as usize] as i32 == B_ST {
                /* right S = cnum[v], left S = cfirst[v]; left pushed last => popped first */
                pda.push(cm.cnum[v as usize]); /* right S */
                pda.push(PDA_STATE);
                pda.push(cm.cfirst[v as usize]); /* left S */
                pda.push(PDA_STATE);
            } else if cm.sttype[v as usize] as i32 != E_ST {
                /* v = cm->nodemap[cm->ndidx[cm->cfirst[v] + cm->cnum[v] - 1]]; */
                let idx = cm.cfirst[v as usize] + cm.cnum[v as usize] - 1;
                v = cm.nodemap[cm.ndidx[idx as usize] as usize];
                pda.push(v);
                pda.push(PDA_STATE);
            }
        } /* end PDA_STATE block */

        /* C reallocates cseq/cstr/ct here when cpos == nalloc; Vec handles it. */
    } /* PDA now empty */

    /* cseq[cpos] = '\0'; cstr[cpos] = '\0';
     * The C null terminators live at index cpos (== clen), outside the
     * 0..clen-1 display range, so we omit them: cseq/cstr have length clen. */

    Some(CMConsensus {
        cseq,
        cstr,
        ct,
        lpos,
        rpos,
        clen: cpos,
    })
}

/// Public: build the consensus and return just `cseq` (the `clen` display
/// chars, 0-based, length == cm->clen). Panics only on the C-equivalent
/// contract failure (no log-odds scores), where C would return NULL.
pub fn create_cm_consensus(cm: &CM) -> Vec<u8> {
    create_cm_consensus_full(cm)
        .expect("CreateCMConsensus: CM lacks log-odds emission scores (esc)")
        .cseq
}

/// cm.c:cm_SetConsensus() (l.2733)
///
/// Fills `cm.consensus` from the consensus display sequence, careful about the
/// off-by-one: `cm.consensus[0]` is a space sentinel, `[1..=clen]` hold the
/// display chars (`cons->cseq[cpos-1]`), and `[clen+1]` is the '\0' terminator.
/// Raises the CMH_CONS flag. Internally constructs the CMConsensus via
/// create_cm_consensus (mirroring the usual caller sequence).
///
/// ```c
/// int
/// cm_SetConsensus(CM_t *cm, CMConsensus_t *cons, ESL_SQ *sq)
/// {
///   ...
///   if (! cm->consensus) ESL_ALLOC(cm->consensus, sizeof(char) * (cm->clen+2));
///   cm->consensus[0] = ' ';
///   for (cpos = 1; cpos <= cm->clen; cpos++) cm->consensus[cpos] = cons->cseq[cpos-1];
///   cm->consensus[cm->clen+1] = '\0';
///   cm->flags |= CMH_CONS;
///   return eslOK;
/// }
/// ```
pub fn cm_set_consensus(cm: &mut CM) {
    let cseq = create_cm_consensus(cm);
    let clen = cm.clen as usize;

    /* allocation: length clen+2. */
    cm.consensus = vec![0u8; clen + 2];

    /* cm->consensus[0] = ' '; */
    cm.consensus[0] = b' ';
    /* for (cpos = 1; cpos <= cm->clen; cpos++) cm->consensus[cpos] = cons->cseq[cpos-1]; */
    for cpos in 1..=clen {
        cm.consensus[cpos] = cseq[cpos - 1];
    }
    /* cm->consensus[cm->clen+1] = '\0'; */
    cm.consensus[clen + 1] = 0;

    /* cm->flags |= CMH_CONS; */
    cm.flags |= CM_CONS;
}
