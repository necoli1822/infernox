//! Faithful from-scratch transcription of Infernal's CP9 HMM layer used by the
//! F6/F7 CM-DP stages (HMM banding). Written directly from the C sources with
//! byte-for-byte fidelity as the goal (see memory `methodology-c-parity-before-speed`
//! and `porting-comment-convention`); it does NOT reference the legacy
//! `cp9.rs`/`cp9_dp.rs` approximation (which stays for the still-live path).
//!
//! Build order (each byte-verified vs C dumps before the next):
//!   (i)   ilogsum integer log-sum table                    [this file, DONE]
//!   (ii)  CP9 model construction from CM (cp9_modelmaker.c) [TODO]
//!   (iii) cp9_Forward (scan)  / cp9_Backward (cp9_dp.c)     [TODO]
//!   (iv)  cp9_FB2HMMBands / cp9_IterateSeq2Bands (hmmband.c)[TODO]

// ---------------------------------------------------------------------------
// (i) ILogsum — integer log-sum, BITS (log2), scaled by INTSCALE
//     (src/logsum.c:72-92, the active version; the second copy at :198 is dead).
// ---------------------------------------------------------------------------

use std::sync::OnceLock;

/// C `#define INFTY 987654321` (infernal.h:157) — the integer-DP -inf sentinel.
pub const INFTY: i32 = 987654321;
/// C `#define INTSCALE 1000.0f` (infernal.h:177).
pub const INTSCALE: f64 = 1000.0;
/// C `LOGSUM_TBL` = 20000 (config.h default, logsum.c:183).
pub const LOGSUM_TBL: usize = 20000;

fn ilogsum_table() -> &'static [i32; LOGSUM_TBL] {
    // C init_ilogsum (logsum.c:74-83):
    //   for (i = 0; i < LOGSUM_TBL; i++)
    //     ilogsum_lookup[i] = rint(INTSCALE * (sreLOG2(1.+sreEXP2((double) -i/INTSCALE))));
    //   sreLOG2(x)=log2(x), sreEXP2(x)=2^x. rint = round-half-to-even.
    static TABLE: OnceLock<Box<[i32; LOGSUM_TBL]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0i32; LOGSUM_TBL]);
        for i in 0..LOGSUM_TBL {
            let v = INTSCALE * (1.0 + 2f64.powf(-(i as f64) / INTSCALE)).log2();
            // rint() rounds ties to even; Rust's round_ties_even matches it.
            t[i] = v.round_ties_even() as i32;
        }
        t
    })
}

/// C `ILogsum(s1, s2)` (logsum.c:86-92):
///   const int max = ESL_MAX(-INFTY, ESL_MAX(s1, s2));
///   const int min = ESL_MIN(s1, s2);
///   return (min <= -INFTY || (max-min) >= LOGSUM_TBL) ? max : max + ilogsum_lookup[max-min];
#[inline]
pub fn ilogsum(s1: i32, s2: i32) -> i32 {
    let max = (-INFTY).max(s1.max(s2));
    let min = s1.min(s2);
    let diff = max - min;
    if min <= -INFTY || diff as i64 >= LOGSUM_TBL as i64 {
        max
    } else {
        max + ilogsum_table()[diff as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilogsum_sentinel_and_bounds() {
        // -INFTY sentinel: ILogsum(-INFTY, s) == s (min <= -INFTY → max).
        assert_eq!(ilogsum(-INFTY, 1234), 1234);
        assert_eq!(ilogsum(5678, -INFTY), 5678);
        // beyond table: (max-min) >= LOGSUM_TBL → max.
        assert_eq!(ilogsum(0, -25000), 0);
        assert_eq!(ilogsum(-25000, 0), 0);
    }

    #[test]
    fn ilogsum_table_matches_c_dump() {
        // Byte-verified against C `DUMP_ILOGSUM` (init_ilogsum, tRNA5.c.cm run).
        let t = ilogsum_table();
        let c: [(usize, i32); 10] = [
            (0, 1000), (1, 1000), (2, 999), (10, 995), (100, 951),
            (693, 695), (1000, 585), (5000, 44), (10000, 1), (19999, 0),
        ];
        for (i, v) in c {
            assert_eq!(t[i], v, "ilogsum_lookup[{}]: got {}, C={}", i, t[i], v);
        }
    }

    #[test]
    fn ilogsum_table_endpoints_match_c_formula() {
        // lookup[0] = rint(1000*log2(1+2^0)) = rint(1000*log2(2)) = 1000.
        let t = ilogsum_table();
        assert_eq!(t[0], 1000);
        // Equal args: ILogsum(s,s) = s + lookup[0] = s + 1000.
        assert_eq!(ilogsum(500, 500), 1500);
        // Non-increasing, decaying to 0 at the tail (t[0]==t[1]==1000).
        assert!(t[2] < t[1] && t[100] < t[2]);
        assert_eq!(t[LOGSUM_TBL - 1], 0);
    }
}

// ---------------------------------------------------------------------------
// (ii-a) CreateEmitMap — consensus column map of the CM (display.c).
//     Assigns each CM node its left/right consensus column (lpos/rpos) and the
//     EL-follows position (epos), via a stack-based DFS. Foundation for
//     CP9_map_cm2hmm. Fresh transcription (no legacy reference).
// ---------------------------------------------------------------------------

use crate::cm::CM;
use crate::constants::{
    BIF_ND, END_ND, MATL_ND, MATP_ND, MATR_ND, BEGL_ND, BEGR_ND, ROOT_ND,
    BEGL_S, BEGR_S, ROOT_S, ROOT_IL, ROOT_IR,
    MATP_MP, MATP_ML, MATP_MR, MATL_ML, MATR_MR,
    MAXCONNECT, NODETYPES, UNIQUESTATES,
    S_ST, IL_ST, IR_ST, E_ST, EL_ST,
    BEGR_IL, BIF_B, END_E, MATL_D, MATL_IL, MATP_D, MATP_IL, MATP_IR, MATR_D, MATR_IR,
};

/// C `CMEmitMap_t` subset used by CP9 mapping: per-node consensus columns.
pub struct EmitMap {
    pub lpos: Vec<i32>, // [0..nodes-1] left consensus column of node
    pub rpos: Vec<i32>, // [0..nodes-1] right consensus column of node
    pub epos: Vec<i32>, // [0..nodes-1] consensus position an EL after this node follows
    pub clen: i32,      // consensus length = rpos[0]-1
}

/// Port of `CreateEmitMap(CM_t *cm)` (display.c). `cm->ndtype[nd]` is the node
/// type; `nodemap[nd]` = first state of node nd; `ndidx[v]` = node of state v;
/// `cfirst[v]`/`cnum[v]` give a BIF's left/right child first states.
pub fn create_emit_map(cm: &CM) -> EmitMap {
    let nodes = cm.nodes as usize;
    // C: lpos[nd]=rpos[nd]=epos[nd]=-1 for all nd.
    let mut lpos = vec![-1i32; nodes];
    let mut rpos = vec![-1i32; nodes];
    let mut epos = vec![-1i32; nodes];

    // C: cpos=0; nd=0; push(0 /*left side*/); push(nd);
    //    while(pop nd) { pop on_right; ... }
    // Stack frames are (on_right, nd); C pushes on_right then nd (nd popped first),
    // so a Rust (on_right, nd) tuple pushed/popped atomically is equivalent.
    let mut cpos: i32 = 0;
    let mut pda: Vec<(i32, i32)> = Vec::new();
    pda.push((0, 0));
    while let Some((on_right, nd)) = pda.pop() {
        let ndi = nd as usize;
        let nt = cm.ndtype[ndi] as i32;
        if on_right != 0 {
            // C: rpos[nd]=cpos+1; if MATP||MATR cpos++
            rpos[ndi] = cpos + 1;
            if nt == MATP_ND || nt == MATR_ND {
                cpos += 1;
            }
        } else {
            // C: if MATP||MATL cpos++; lpos[nd]=cpos;
            if nt == MATP_ND || nt == MATL_ND {
                cpos += 1;
            }
            lpos[ndi] = cpos;
            if nt == BIF_ND {
                // C: push self for right side, then right child, then left child.
                //    right child = ndidx[cnum[nodemap[nd]]], left = ndidx[cfirst[nodemap[nd]]]
                pda.push((1, nd));
                let nm = cm.nodemap[ndi] as usize;
                let rchild = cm.ndidx[cm.cnum[nm] as usize];
                let lchild = cm.ndidx[cm.cfirst[nm] as usize];
                pda.push((0, rchild));
                pda.push((0, lchild));
            } else {
                // C: push self for right side; if not END, push child nd+1.
                pda.push((1, nd));
                if nt != END_ND {
                    pda.push((0, nd + 1));
                }
            }
        }
    }

    // C: epos map, nd from nodes-1 down to 0. `cpos` carries over between
    //    iterations; reset at END (lpos[nd]) and BIF (epos[right child]).
    for nd in (0..nodes).rev() {
        let nt = cm.ndtype[nd] as i32;
        if nt == END_ND {
            cpos = lpos[nd];
        } else if nt == BIF_ND {
            let nm = cm.nodemap[nd] as usize;
            cpos = epos[cm.ndidx[cm.cnum[nm] as usize] as usize];
        }
        epos[nd] = cpos;
    }

    let clen = rpos[0] - 1;
    EmitMap { lpos, rpos, epos, clen }
}

// ---------------------------------------------------------------------------
// (ii-b) CP9_map_cm2hmm — bidirectional CM-state <-> HMM-node/state map
//     (cp9_modelmaker.c:343-744, incl. AllocCP9Map:59 and map_helper:713).
//     Fresh transcription. State-within-node offsets from cm->nodemap[n]:
//       MATP: MP=0 ML=1 MR=2 D=3 IL=4 IR=5 | MATL: ML=0 D=1 IL=2
//       MATR: MR=0 D=1 IR=2 | BEGR: S=0 IL=1 | ROOT: S=0 IL=1 IR=2
// ---------------------------------------------------------------------------

use crate::constants::BIF_ND as _BIF;

/// C `CP9Map_t` (struct cp9map_s). Bidirectional CM<->CP9 maps.
pub struct CP9Map {
    pub hmm_m: i32,             // consensus length (HMM node count)
    pub cm_m: i32,
    pub cm_nodes: i32,
    pub nd2lpos: Vec<i32>,      // [node] left consensus col (MATP/MATL), else -1
    pub nd2rpos: Vec<i32>,      // [node] right consensus col (MATP/MATR), else -1
    pub pos2nd: Vec<i32>,       // [0..hmm_m] consensus col -> CM node
    pub hns2cs: Vec<[[i32; 2]; 3]>, // [0..hmm_m][ks=M/I/D][0..1] -> CM state(s)
    pub cs2hn: Vec<[i32; 2]>,   // [0..cm_m] CM state -> HMM node(s)
    pub cs2hs: Vec<[i32; 2]>,   // [0..cm_m] CM state -> HMM state(s)
}

/// C sizes state arrays to `cm->M+1` with the extra slot at index `cm->M` being
/// the EL (end-local) state (`sttype[M]=EL_st`, cm.c:271). Rust's CM stores only
/// the `cm.m` real states (0..cm.m-1), so these accessors faithfully reproduce
/// C's oversized arrays: index `cm.m` (== C's cm->M) is the EL state.
#[inline]
fn stid_at(cm: &CM, v: usize) -> i32 {
    if v < cm.stid.len() { cm.stid[v] as i32 } else { -1 } // EL: not a match state
}
#[inline]
fn sttype_at(cm: &CM, v: usize) -> i32 {
    if v < cm.sttype.len() { cm.sttype[v] as i32 } else { EL_ST }
}

/// C map_helper (cp9_modelmaker.c:713): register the (k,ks)<->v mapping,
/// refusing to map a detached insert (v whose v+1 is an E state).
fn map_helper(cm: &CM, m: &mut CP9Map, k: i32, ks: i32, v: i32) {
    // C: if(ks==1 && cm->sttype[v+1]==E_st) return;  (detached insert)
    if ks == 1 && sttype_at(cm, (v + 1) as usize) == E_ST {
        return;
    }
    let vu = v as usize;
    if m.cs2hn[vu][0] == -1 {
        m.cs2hn[vu][0] = k;
        m.cs2hs[vu][0] = ks;
    } else if m.cs2hn[vu][1] == -1 {
        m.cs2hn[vu][1] = k;
        m.cs2hs[vu][1] = ks;
    } else {
        panic!("map_helper: cs2hn[{}][1] already set", v);
    }
    let ku = k as usize;
    let ksu = ks as usize;
    if m.hns2cs[ku][ksu][0] == -1 {
        m.hns2cs[ku][ksu][0] = v;
    } else if m.hns2cs[ku][ksu][1] == -1 {
        m.hns2cs[ku][ksu][1] = v;
    } else {
        panic!("map_helper: hns2cs[{}][{}][1] already set", k, ks);
    }
}

/// Port of `CP9_map_cm2hmm` (cp9_modelmaker.c:343). Requires node types from
/// `cm` and the consensus columns from `create_emit_map`.
pub fn cp9_map_cm2hmm(cm: &CM) -> CP9Map {
    // AllocCP9Map (:59): hmm_M = 2 per MATP_MP + 1 per MATL_ML/MATR_MR.
    let mut hmm_m = 0i32;
    for v in 0..=cm.m as usize {
        let id = stid_at(cm, v);
        if id == MATP_MP {
            hmm_m += 2;
        } else if id == MATL_ML || id == MATR_MR {
            hmm_m += 1;
        }
    }
    let cm_nodes = cm.nodes;
    let mut m = CP9Map {
        hmm_m,
        cm_m: cm.m,
        cm_nodes,
        nd2lpos: vec![-1; cm_nodes as usize],
        nd2rpos: vec![-1; cm_nodes as usize],
        pos2nd: vec![-1; (hmm_m + 1) as usize],
        hns2cs: vec![[[-1; 2]; 3]; (hmm_m + 1) as usize],
        cs2hn: vec![[-1; 2]; (cm.m + 1) as usize],
        cs2hs: vec![[-1; 2]; (cm.m + 1) as usize],
    };

    // C:362-383 copy emit-map lpos/rpos for MATP/MATL (lpos), MATP/MATR (rpos),
    //   and pos2nd[col]=node.
    let emap = create_emit_map(cm);
    for n in 0..cm.nodes as usize {
        let nt = cm.ndtype[n] as i32;
        if nt == MATP_ND || nt == MATL_ND {
            m.nd2lpos[n] = emap.lpos[n];
            m.pos2nd[emap.lpos[n] as usize] = n as i32;
        }
        if nt == MATP_ND || nt == MATR_ND {
            m.nd2rpos[n] = emap.rpos[n];
            m.pos2nd[emap.rpos[n] as usize] = n as i32;
        }
    }

    // C:385-397 HMM node k=0 (ROOT): match<-ROOT_S(v0), insert<-ROOT_IL(v1).
    map_helper(cm, &mut m, 0, 0, 0); // ROOT_S
    map_helper(cm, &mut m, 0, 1, 1); // ROOT_IL
    // ROOT_IR handled where the last column's right-insert maps (below).

    // C:400-621 step through HMM nodes 1..hmm_M.
    for k in 1..=hmm_m {
        let n = m.pos2nd[k as usize];
        let nu = n as usize;
        let is_left = m.nd2lpos[nu] == k;
        let is_right = m.nd2rpos[nu] == k;
        let nt = cm.ndtype[nu] as i32;
        let nm = cm.nodemap[nu]; // first state of node n

        if nt == MATP_ND {
            if is_left {
                // C:424-442
                map_helper(cm, &mut m, k, 0, nm);     // MATP_MP (match)
                map_helper(cm, &mut m, k, 0, nm + 1); // MATP_ML (match)
                map_helper(cm, &mut m, k, 1, nm + 4); // MATP_IL (insert)
                map_helper(cm, &mut m, k, 2, nm + 2); // MATP_MR (delete)
                map_helper(cm, &mut m, k, 2, nm + 3); // MATP_D  (delete)
            } else if is_right {
                // C:443-534
                map_helper(cm, &mut m, k, 0, nm);     // MATP_MP (match)
                map_helper(cm, &mut m, k, 0, nm + 2); // MATP_MR (match)
                // insert to the RIGHT of column k: find CM state modelling k+1.
                if k != hmm_m {
                    let nn = m.pos2nd[(k + 1) as usize];
                    let nnu = nn as usize;
                    if m.nd2lpos[nnu] == k + 1 {
                        // find closest BEGR node above nn
                        let mut n_begr = nn;
                        while n_begr >= 0 && cm.ndtype[n_begr as usize] as i32 != BEGR_ND {
                            n_begr -= 1;
                        }
                        map_helper(cm, &mut m, k, 1, cm.nodemap[n_begr as usize] + 1); // BEGR_IL
                    } else if m.nd2rpos[nnu] == k + 1 {
                        let nnt = cm.ndtype[nnu] as i32;
                        if nnt == MATP_ND {
                            map_helper(cm, &mut m, k, 1, cm.nodemap[nnu] + 5); // MATP_IR
                        } else if nnt == MATR_ND {
                            map_helper(cm, &mut m, k, 1, cm.nodemap[nnu] + 2); // MATR_IR
                        }
                    }
                } else {
                    map_helper(cm, &mut m, k, 1, 2); // ROOT_IR
                }
                // MATP_IR lookback: if column k-1 is modelled by a MATP left half
                // (C:498-526), map MATP_IR to HMM node k-1 insert.
                let pn_km1 = m.pos2nd[(k - 1) as usize] as usize;
                if m.nd2lpos[pn_km1] == k - 1 {
                    let km1nt = cm.ndtype[pn_km1] as i32;
                    if km1nt != MATL_ND && km1nt != MATP_ND {
                        panic!("cp9_map_cm2hmm: unexpected node type at k-1 (case 0)");
                    }
                    map_helper(cm, &mut m, k - 1, 1, nm + 5); // MATP_IR
                }
                // C:528-533 delete
                map_helper(cm, &mut m, k, 2, nm + 1); // MATP_ML (delete)
                map_helper(cm, &mut m, k, 2, nm + 3); // MATP_D  (delete)
            }
        } else if nt == MATL_ND {
            // C:537-557
            map_helper(cm, &mut m, k, 0, nm);     // MATL_ML (match)
            map_helper(cm, &mut m, k, 1, nm + 2); // MATL_IL (insert)
            map_helper(cm, &mut m, k, 2, nm + 1); // MATL_D  (delete)
            if k == hmm_m {
                map_helper(cm, &mut m, k, 1, 2); // ROOT_IR (last column insert)
            }
        } else if nt == MATR_ND {
            // C:559-619
            map_helper(cm, &mut m, k, 0, nm); // MATR_MR (match)
            if k != hmm_m {
                let nn = m.pos2nd[(k + 1) as usize];
                let nnu = nn as usize;
                if m.nd2lpos[nnu] == k + 1 {
                    let mut n_begr = nn;
                    while cm.ndtype[n_begr as usize] as i32 != BEGR_ND && n_begr >= 0 {
                        n_begr -= 1;
                    }
                    map_helper(cm, &mut m, k, 1, cm.nodemap[n_begr as usize] + 1); // BEGR_IL
                } else if m.nd2rpos[nnu] == k + 1 {
                    let nnt = cm.ndtype[nnu] as i32;
                    if nnt == MATP_ND {
                        map_helper(cm, &mut m, k, 1, cm.nodemap[nnu] + 5); // MATP_IR
                    } else if nnt == MATR_ND {
                        map_helper(cm, &mut m, k, 1, cm.nodemap[nnu] + 2); // MATR_IR
                    }
                }
            } else {
                map_helper(cm, &mut m, k, 1, 2); // ROOT_IR
            }
            let pn_km1 = m.pos2nd[(k - 1) as usize] as usize;
            if m.nd2lpos[pn_km1] == k - 1 {
                panic!("cp9_map_cm2hmm: unexpected MATL above MATR (case 1)");
            }
            map_helper(cm, &mut m, k, 2, nm + 1); // MATR_D (delete)
        } else {
            panic!("cp9_map_cm2hmm: HMM node {} maps to non-MAT node type {}", k, nt);
        }
    }

    // Suppress unused-const warnings for symmetry imports.
    let _ = (IL_ST, IR_ST, ROOT_ND, _BIF);
    m
}

// =============================================================================
// (ii-c) cm_ExpectedStateOccupancy (psi) — cm.c:4250
// psi[v] = expected number of times CM state v is entered in a globally
// configured CM. Uses a transition map (cm_CreateTransitionMap, cm.c:4495).
// =============================================================================

const CMH_LOCAL_END: u32 = 1 << 11; // infernal.h:1938

/// C: TotalStatesInNode (cm.c:1107)
fn total_states_in_node(ndtype: i32) -> i32 {
    match ndtype {
        x if x == _BIF => 1,   // BIF_nd
        x if x == MATP_ND => 6,
        x if x == MATL_ND => 3,
        x if x == MATR_ND => 3,
        x if x == BEGL_ND => 1,
        x if x == BEGR_ND => 2,
        x if x == ROOT_ND => 3,
        x if x == END_ND => 1,
        _ => panic!("Bogus node type {}", ndtype),
    }
}

/// C: StateIsDetached (cm.c:1775): TRUE iff stid[v+1]==END_E.
fn state_is_detached(cm: &CM, v: usize) -> bool {
    stid_at(cm, v + 1) == END_E
}

/// C: esl_vec_FSum (Kahan summation, esl_vectorops.c).
fn esl_vec_fsum(vec: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for &vi in vec {
        let y = vi - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// C: esl_vec_FNorm (esl_vectorops.c:1152).
fn esl_vec_fnorm(vec: &mut [f32]) {
    let sum = esl_vec_fsum(vec);
    let n = vec.len();
    if sum != 0.0 {
        for x in vec.iter_mut() {
            *x /= sum;
        }
    } else {
        for x in vec.iter_mut() {
            *x = 1.0 / n as f32;
        }
    }
}

/// C: cm_CreateTransitionMap (cm.c:4495). tmap[UNIQUESTATES][NODETYPES][UNIQUESTATES],
/// -1 = invalid. Table generated 1:1 from the C source.
pub fn cm_create_transition_map() -> Vec<Vec<Vec<i8>>> {
    let us = UNIQUESTATES as usize;
    let nt = NODETYPES as usize;
    let mut tmap = vec![vec![vec![-1i8; us]; nt]; us];
      tmap[ROOT_S as usize][BIF_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_S as usize][BIF_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_S as usize][BIF_ND as usize][BIF_B as usize] = 2;
      tmap[ROOT_S as usize][MATP_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_S as usize][MATP_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_S as usize][MATP_ND as usize][MATP_MP as usize] = 2;
      tmap[ROOT_S as usize][MATP_ND as usize][MATP_ML as usize] = 3;
      tmap[ROOT_S as usize][MATP_ND as usize][MATP_MR as usize] = 4;
      tmap[ROOT_S as usize][MATP_ND as usize][MATP_D as usize] = 5;
      tmap[ROOT_S as usize][MATL_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_S as usize][MATL_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_S as usize][MATL_ND as usize][MATL_ML as usize] = 2;
      tmap[ROOT_S as usize][MATL_ND as usize][MATL_D as usize] = 3;
      tmap[ROOT_S as usize][MATR_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_S as usize][MATR_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_S as usize][MATR_ND as usize][MATR_MR as usize] = 2;
      tmap[ROOT_S as usize][MATR_ND as usize][MATR_D as usize] = 3;
      tmap[ROOT_IL as usize][BIF_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_IL as usize][BIF_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_IL as usize][BIF_ND as usize][BIF_B as usize] = 2;
      tmap[ROOT_IL as usize][MATP_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_IL as usize][MATP_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_IL as usize][MATP_ND as usize][MATP_MP as usize] = 2;
      tmap[ROOT_IL as usize][MATP_ND as usize][MATP_ML as usize] = 3;
      tmap[ROOT_IL as usize][MATP_ND as usize][MATP_MR as usize] = 4;
      tmap[ROOT_IL as usize][MATP_ND as usize][MATP_D as usize] = 5;
      tmap[ROOT_IL as usize][MATL_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_IL as usize][MATL_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_IL as usize][MATL_ND as usize][MATL_ML as usize] = 2;
      tmap[ROOT_IL as usize][MATL_ND as usize][MATL_D as usize] = 3;
      tmap[ROOT_IL as usize][MATR_ND as usize][ROOT_IL as usize] = 0;
      tmap[ROOT_IL as usize][MATR_ND as usize][ROOT_IR as usize] = 1;
      tmap[ROOT_IL as usize][MATR_ND as usize][MATR_MR as usize] = 2;
      tmap[ROOT_IL as usize][MATR_ND as usize][MATR_D as usize] = 3;
      tmap[ROOT_IR as usize][BIF_ND as usize][ROOT_IR as usize] = 0;
      tmap[ROOT_IR as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[ROOT_IR as usize][MATP_ND as usize][ROOT_IR as usize] = 0;
      tmap[ROOT_IR as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[ROOT_IR as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[ROOT_IR as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[ROOT_IR as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[ROOT_IR as usize][MATL_ND as usize][ROOT_IR as usize] = 0;
      tmap[ROOT_IR as usize][MATL_ND as usize][MATL_ML as usize] = 1;
      tmap[ROOT_IR as usize][MATL_ND as usize][MATL_D as usize] = 2;
      tmap[ROOT_IR as usize][MATR_ND as usize][ROOT_IR as usize] = 0;
      tmap[ROOT_IR as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[ROOT_IR as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[BEGL_S as usize][BIF_ND as usize][BIF_B as usize] = 0;
      tmap[BEGL_S as usize][MATP_ND as usize][MATP_MP as usize] = 0;
      tmap[BEGL_S as usize][MATP_ND as usize][MATP_ML as usize] = 1;
      tmap[BEGL_S as usize][MATP_ND as usize][MATP_MR as usize] = 2;
      tmap[BEGL_S as usize][MATP_ND as usize][MATP_D as usize] = 3;
      tmap[BEGR_S as usize][BIF_ND as usize][BEGR_IL as usize] = 0;
      tmap[BEGR_S as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[BEGR_S as usize][MATP_ND as usize][BEGR_IL as usize] = 0;
      tmap[BEGR_S as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[BEGR_S as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[BEGR_S as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[BEGR_S as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[BEGR_S as usize][MATL_ND as usize][BEGR_IL as usize] = 0;
      tmap[BEGR_S as usize][MATL_ND as usize][MATL_ML as usize] = 1;
      tmap[BEGR_S as usize][MATL_ND as usize][MATL_D as usize] = 2;
      tmap[BEGR_IL as usize][BIF_ND as usize][BEGR_IL as usize] = 0;
      tmap[BEGR_IL as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[BEGR_IL as usize][MATP_ND as usize][BEGR_IL as usize] = 0;
      tmap[BEGR_IL as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[BEGR_IL as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[BEGR_IL as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[BEGR_IL as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[BEGR_IL as usize][MATL_ND as usize][BEGR_IL as usize] = 0;
      tmap[BEGR_IL as usize][MATL_ND as usize][MATL_ML as usize] = 1;
      tmap[BEGR_IL as usize][MATL_ND as usize][MATL_D as usize] = 2;
      tmap[MATP_MP as usize][BIF_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MP as usize][BIF_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MP as usize][BIF_ND as usize][BIF_B as usize] = 2;
      tmap[MATP_MP as usize][MATP_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MP as usize][MATP_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MP as usize][MATP_ND as usize][MATP_MP as usize] = 2;
      tmap[MATP_MP as usize][MATP_ND as usize][MATP_ML as usize] = 3;
      tmap[MATP_MP as usize][MATP_ND as usize][MATP_MR as usize] = 4;
      tmap[MATP_MP as usize][MATP_ND as usize][MATP_D as usize] = 5;
      tmap[MATP_MP as usize][MATL_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MP as usize][MATL_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MP as usize][MATL_ND as usize][MATL_ML as usize] = 2;
      tmap[MATP_MP as usize][MATL_ND as usize][MATL_D as usize] = 3;
      tmap[MATP_MP as usize][MATR_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MP as usize][MATR_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MP as usize][MATR_ND as usize][MATR_MR as usize] = 2;
      tmap[MATP_MP as usize][MATR_ND as usize][MATR_D as usize] = 3;
      tmap[MATP_MP as usize][END_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MP as usize][END_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MP as usize][END_ND as usize][END_E as usize] = 2;
      tmap[MATP_ML as usize][BIF_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_ML as usize][BIF_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_ML as usize][BIF_ND as usize][BIF_B as usize] = 2;
      tmap[MATP_ML as usize][MATP_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_ML as usize][MATP_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_ML as usize][MATP_ND as usize][MATP_MP as usize] = 2;
      tmap[MATP_ML as usize][MATP_ND as usize][MATP_ML as usize] = 3;
      tmap[MATP_ML as usize][MATP_ND as usize][MATP_MR as usize] = 4;
      tmap[MATP_ML as usize][MATP_ND as usize][MATP_D as usize] = 5;
      tmap[MATP_ML as usize][MATL_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_ML as usize][MATL_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_ML as usize][MATL_ND as usize][MATL_ML as usize] = 2;
      tmap[MATP_ML as usize][MATL_ND as usize][MATL_D as usize] = 3;
      tmap[MATP_ML as usize][MATR_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_ML as usize][MATR_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_ML as usize][MATR_ND as usize][MATR_MR as usize] = 2;
      tmap[MATP_ML as usize][MATR_ND as usize][MATR_D as usize] = 3;
      tmap[MATP_ML as usize][END_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_ML as usize][END_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_ML as usize][END_ND as usize][END_E as usize] = 2;
      tmap[MATP_MR as usize][BIF_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MR as usize][BIF_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MR as usize][BIF_ND as usize][BIF_B as usize] = 2;
      tmap[MATP_MR as usize][MATP_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MR as usize][MATP_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MR as usize][MATP_ND as usize][MATP_MP as usize] = 2;
      tmap[MATP_MR as usize][MATP_ND as usize][MATP_ML as usize] = 3;
      tmap[MATP_MR as usize][MATP_ND as usize][MATP_MR as usize] = 4;
      tmap[MATP_MR as usize][MATP_ND as usize][MATP_D as usize] = 5;
      tmap[MATP_MR as usize][MATL_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MR as usize][MATL_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MR as usize][MATL_ND as usize][MATL_ML as usize] = 2;
      tmap[MATP_MR as usize][MATL_ND as usize][MATL_D as usize] = 3;
      tmap[MATP_MR as usize][MATR_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MR as usize][MATR_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MR as usize][MATR_ND as usize][MATR_MR as usize] = 2;
      tmap[MATP_MR as usize][MATR_ND as usize][MATR_D as usize] = 3;
      tmap[MATP_MR as usize][END_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_MR as usize][END_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_MR as usize][END_ND as usize][END_E as usize] = 2;
      tmap[MATP_D as usize][BIF_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_D as usize][BIF_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_D as usize][BIF_ND as usize][BIF_B as usize] = 2;
      tmap[MATP_D as usize][MATP_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_D as usize][MATP_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_D as usize][MATP_ND as usize][MATP_MP as usize] = 2;
      tmap[MATP_D as usize][MATP_ND as usize][MATP_ML as usize] = 3;
      tmap[MATP_D as usize][MATP_ND as usize][MATP_MR as usize] = 4;
      tmap[MATP_D as usize][MATP_ND as usize][MATP_D as usize] = 5;
      tmap[MATP_D as usize][MATL_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_D as usize][MATL_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_D as usize][MATL_ND as usize][MATL_ML as usize] = 2;
      tmap[MATP_D as usize][MATL_ND as usize][MATL_D as usize] = 3;
      tmap[MATP_D as usize][MATR_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_D as usize][MATR_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_D as usize][MATR_ND as usize][MATR_MR as usize] = 2;
      tmap[MATP_D as usize][MATR_ND as usize][MATR_D as usize] = 3;
      tmap[MATP_D as usize][END_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_D as usize][END_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_D as usize][END_ND as usize][END_E as usize] = 2;
      tmap[MATP_IL as usize][BIF_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_IL as usize][BIF_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_IL as usize][BIF_ND as usize][BIF_B as usize] = 2;
      tmap[MATP_IL as usize][MATP_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_IL as usize][MATP_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_IL as usize][MATP_ND as usize][MATP_MP as usize] = 2;
      tmap[MATP_IL as usize][MATP_ND as usize][MATP_ML as usize] = 3;
      tmap[MATP_IL as usize][MATP_ND as usize][MATP_MR as usize] = 4;
      tmap[MATP_IL as usize][MATP_ND as usize][MATP_D as usize] = 5;
      tmap[MATP_IL as usize][MATL_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_IL as usize][MATL_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_IL as usize][MATL_ND as usize][MATL_ML as usize] = 2;
      tmap[MATP_IL as usize][MATL_ND as usize][MATL_D as usize] = 3;
      tmap[MATP_IL as usize][MATR_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_IL as usize][MATR_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_IL as usize][MATR_ND as usize][MATR_MR as usize] = 2;
      tmap[MATP_IL as usize][MATR_ND as usize][MATR_D as usize] = 3;
      tmap[MATP_IL as usize][END_ND as usize][MATP_IL as usize] = 0;
      tmap[MATP_IL as usize][END_ND as usize][MATP_IR as usize] = 1;
      tmap[MATP_IL as usize][END_ND as usize][END_E as usize] = 2;
      tmap[MATP_IR as usize][BIF_ND as usize][MATP_IR as usize] = 0;
      tmap[MATP_IR as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[MATP_IR as usize][MATP_ND as usize][MATP_IR as usize] = 0;
      tmap[MATP_IR as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[MATP_IR as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[MATP_IR as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[MATP_IR as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[MATP_IR as usize][MATL_ND as usize][MATP_IR as usize] = 0;
      tmap[MATP_IR as usize][MATL_ND as usize][MATL_ML as usize] = 1;
      tmap[MATP_IR as usize][MATL_ND as usize][MATL_D as usize] = 2;
      tmap[MATP_IR as usize][MATR_ND as usize][MATP_IR as usize] = 0;
      tmap[MATP_IR as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[MATP_IR as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[MATP_IR as usize][END_ND as usize][MATP_IR as usize] = 0;
      tmap[MATP_IR as usize][END_ND as usize][END_E as usize] = 1;
      tmap[MATL_ML as usize][BIF_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_ML as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[MATL_ML as usize][MATP_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_ML as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[MATL_ML as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[MATL_ML as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[MATL_ML as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[MATL_ML as usize][MATL_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_ML as usize][MATL_ND as usize][MATL_ML as usize] = 1;
      tmap[MATL_ML as usize][MATL_ND as usize][MATL_D as usize] = 2;
      tmap[MATL_ML as usize][MATR_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_ML as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[MATL_ML as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[MATL_ML as usize][END_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_ML as usize][END_ND as usize][END_E as usize] = 1;
      tmap[MATL_D as usize][BIF_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_D as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[MATL_D as usize][MATP_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_D as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[MATL_D as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[MATL_D as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[MATL_D as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[MATL_D as usize][MATL_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_D as usize][MATL_ND as usize][MATL_ML as usize] = 1;
      tmap[MATL_D as usize][MATL_ND as usize][MATL_D as usize] = 2;
      tmap[MATL_D as usize][MATR_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_D as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[MATL_D as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[MATL_D as usize][END_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_D as usize][END_ND as usize][END_E as usize] = 1;
      tmap[MATL_IL as usize][BIF_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_IL as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[MATL_IL as usize][MATP_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_IL as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[MATL_IL as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[MATL_IL as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[MATL_IL as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[MATL_IL as usize][MATL_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_IL as usize][MATL_ND as usize][MATL_ML as usize] = 1;
      tmap[MATL_IL as usize][MATL_ND as usize][MATL_D as usize] = 2;
      tmap[MATL_IL as usize][MATR_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_IL as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[MATL_IL as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[MATL_IL as usize][END_ND as usize][MATL_IL as usize] = 0;
      tmap[MATL_IL as usize][END_ND as usize][END_E as usize] = 1;
      tmap[MATR_MR as usize][BIF_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_MR as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[MATR_MR as usize][MATP_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_MR as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[MATR_MR as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[MATR_MR as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[MATR_MR as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[MATR_MR as usize][MATR_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_MR as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[MATR_MR as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[MATR_D as usize][BIF_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_D as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[MATR_D as usize][MATP_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_D as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[MATR_D as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[MATR_D as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[MATR_D as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[MATR_D as usize][MATR_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_D as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[MATR_D as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[MATR_IR as usize][BIF_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_IR as usize][BIF_ND as usize][BIF_B as usize] = 1;
      tmap[MATR_IR as usize][MATP_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_IR as usize][MATP_ND as usize][MATP_MP as usize] = 1;
      tmap[MATR_IR as usize][MATP_ND as usize][MATP_ML as usize] = 2;
      tmap[MATR_IR as usize][MATP_ND as usize][MATP_MR as usize] = 3;
      tmap[MATR_IR as usize][MATP_ND as usize][MATP_D as usize] = 4;
      tmap[MATR_IR as usize][MATR_ND as usize][MATR_IR as usize] = 0;
      tmap[MATR_IR as usize][MATR_ND as usize][MATR_MR as usize] = 1;
      tmap[MATR_IR as usize][MATR_ND as usize][MATR_D as usize] = 2;
      tmap[BIF_B as usize][BEGL_ND as usize][BEGL_S as usize] = 0;
      tmap[BIF_B as usize][BEGR_ND as usize][BEGR_S as usize] = 0;
    tmap
}

/// C: cm_ExpectedStateOccupancy (cm.c:4250). Returns psi[0..cm.m-1].
pub fn cm_expected_state_occupancy(cm: &CM) -> Vec<f64> {
    let m = cm.m as usize;
    // tol: 0.001 unless clen > 25000 (cm.c:4272)
    let mut _tol = 0.001_f64;
    if cm.clen > 25000 {
        _tol = (cm.clen as f32 as f64 / 25000.0) * 0.001;
    }

    // make a copy of the CM transitions (cm.c:4277-4282)
    let mut t_copy: Vec<[f32; MAXCONNECT as usize]> = vec![[0.0; MAXCONNECT as usize]; m];
    for v in 0..m {
        for k in 0..MAXCONNECT as usize {
            t_copy[v][k] = cm.t[v][k];
        }
    }
    // local begins: redefine transitions out of state 0 (cm.c:4284)
    if let Some(rt) = &cm.root_trans {
        let n = cm.cnum[0] as usize;
        for k in 0..n {
            t_copy[0][k] = rt[k];
        }
    }
    // local ends: renormalize transitions out of each non-END-preceding MAT*/BEG* state (cm.c:4290)
    if cm.flags & CMH_LOCAL_END != 0 {
        for nd in 1..cm.nodes as usize {
            let ndt = cm.ndtype[nd] as i32;
            if (ndt == MATP_ND || ndt == MATL_ND || ndt == MATR_ND || ndt == BEGL_ND || ndt == BEGR_ND)
                && cm.ndtype[nd + 1] as i32 != END_ND
            {
                let v = cm.nodemap[nd] as usize;
                let cnum = cm.cnum[v] as usize;
                esl_vec_fnorm(&mut t_copy[v][0..cnum]);
            }
        }
    }

    let mut psi = vec![0.0f64; m];
    let tmap = cm_create_transition_map();

    // psi[v] = expected number of times state v is entered (cm.c:4311)
    for v in 0..m {
        let is_insert = if cm.sttype[v] as i32 == IL_ST || cm.sttype[v] as i32 == IR_ST {
            1
        } else {
            0
        };
        if cm.sttype[v] as i32 == S_ST {
            psi[v] = 1.0; // start states are visited in every parse
        } else {
            let final_y = is_insert; // insert self loops handled separately
            let mut y = cm.pnum[v] - 1;
            while y >= final_y {
                let x = (cm.plast[v] - y) as usize; // a parent of v
                let tmap_val =
                    tmap[cm.stid[x] as usize][cm.ndtype[(cm.ndidx[v] + is_insert) as usize] as usize][cm.stid[v] as usize];
                psi[v] += psi[x] * t_copy[x][tmap_val as usize] as f64;
                y -= 1;
            }
            if is_insert == 1 {
                // contribution of the self insertion loops (cm.c:4325)
                psi[v] += psi[v] * (t_copy[v][0] as f64 / (1.0 - t_copy[v][0] as f64));
            }
        }
    }

    // sanity check: split-set psi per node ~ 1.0 (cm.c:4332)
    for nd in 0..cm.nodes as usize {
        let mut summed_psi = 0.0f64;
        let nstates = total_states_in_node(cm.ndtype[nd] as i32);
        let start = cm.nodemap[nd] as usize;
        for v in start..(start + nstates as usize) {
            if cm.sttype[v] as i32 != IL_ST && cm.sttype[v] as i32 != IR_ST {
                summed_psi += psi[v];
            }
        }
        if summed_psi < (1.0 - _tol) || summed_psi > (1.0 + _tol) {
            panic!("summed psi of split states in node {} not 1.0 but : {}", nd, summed_psi);
        }
    }
    // sanity check: only detached inserts may have psi==0 (cm.c:4345)
    for v in 0..m {
        if psi[v] == 0.0 && !state_is_detached(cm, v) {
            panic!("psi of state v:{} is 0.0 and this state is not a detached insert!", v);
        }
    }

    let _ = (BEGR_IL, BIF_B, MATL_D, MATL_IL, MATP_D, MATP_IL, MATP_IR, MATR_D, MATR_IR, END_E, S_ST);
    psi
}

// =============================================================================
// (ii-d) CP9 emissions — build_cp9_hmm emission block (cp9_modelmaker.c:262-307)
// =============================================================================

use crate::cm::{ALPHABET_SIZE, ALPHABET_SIZE_P};

const HMMMATCH: usize = 0; // infernal.h:703
const HMMINSERT: usize = 1; // infernal.h:704
const HMMDELETE: usize = 2; // infernal.h:705

// CP9 transition indices (infernal.h:665-675), cp9_NTRANS=10
const CTMM: usize = 0;
const CTMI: usize = 1;
const CTMD: usize = 2;
const _CTMEL: usize = 3;
const CTIM: usize = 4;
const CTII: usize = 5;
const CTID: usize = 6;
const CTDM: usize = 7;
const CTDI: usize = 8;
const CTDD: usize = 9;
const CP9_NTRANS: usize = 10;
const CP9_TRANS_INSERT_OFFSET: usize = 4; // infernal.h:678
const CP9_TRANS_DELETE_OFFSET: usize = 7; // infernal.h:679

/// CM Plan 9 HMM (grown incrementally during faithful construction).
/// Indices run 0..=M (node 0 is the special ROOT/N node).
pub struct CP9 {
    pub m: i32,
    pub mat: Vec<[f32; ALPHABET_SIZE]>, // match emissions, [0..=M]
    pub ins: Vec<[f32; ALPHABET_SIZE]>, // insert emissions, [0..=M]
    pub t: Vec<[f32; CP9_NTRANS]>,      // transitions, [0..=M]
    pub begin: Vec<f32>,                // begin[k] = B->M_k, [0..=M] (1..=M used)
    pub end: Vec<f32>,                  // end[k] = M_k->E, [0..=M] (1..=M used)
    pub null: [f32; ALPHABET_SIZE],     // null model (CPlan9SetNullModel: = cm.null, p1=1.0)
    pub flags: u32,                     // CPLAN9_* flags
    // EL (end-local) config (CPlan9InitEL):
    pub has_el: Vec<bool>, // [0..=M]
    pub el_self: f32,
    pub el_selfsc: i32,
    // EL DP connectivity (CPlan9InitEL 2nd pass), sized [0..=M+1]:
    pub el_from_ct: Vec<i32>,       // # EL states that can transit into node k
    pub el_from_idx: Vec<Vec<i32>>, // el_from_idx[k] = list of source HMM nodes (lpos)
    // Integer log-odds scores (CP9Logoddsify). msc/isc indexed [k][x] for the
    // full extended alphabet x in 0..Kp: canonical (0..K), gap(K)/nonres(Kp-2)/
    // missing(Kp-1) = -INFTY, degenerate (K+1..Kp-3) = null-weighted IExpectScore.
    pub msc: Vec<[i32; ALPHABET_SIZE_P]>, // [0..=M]
    pub isc: Vec<[i32; ALPHABET_SIZE_P]>, // [0..=M]
    pub tsc: Vec<[i32; CP9_NTRANS]>,    // [0..=M]
    pub bsc: Vec<i32>,                  // [0..=M]
    pub esc: Vec<i32>,                  // [0..=M]
    pub otsc: Vec<[i32; CP9O_NTRANS]>,  // [0..=M] reordered for DP
}

// CPLAN9 flags (infernal.h:655-659)
const CPLAN9_HASBITS: u32 = 1 << 0;
const CPLAN9_HASPROB: u32 = 1 << 1;
const CPLAN9_LOCAL_BEGIN: u32 = 1 << 2;
const CPLAN9_LOCAL_END: u32 = 1 << 3;
const CPLAN9_EL: u32 = 1 << 4;

// cp9O reordered transition indices (infernal.h:685-700), cp9O_NTRANS=12
const CP9O_NTRANS: usize = 12;
const CP9O_MM: usize = 0;
const CP9O_IM: usize = 1;
const CP9O_DM: usize = 2;
const CP9O_BM: usize = 3;
const CP9O_MI: usize = 4;
const CP9O_II: usize = 5;
const CP9O_DI: usize = 6;
const CP9O_MD: usize = 7;
const CP9O_ID: usize = 8;
const CP9O_DD: usize = 9;
const CP9O_ME: usize = 10;
const CP9O_MEL: usize = 11;

const CP9_INFTY: i32 = 987654321; // logsum.c INFTY

impl CP9 {
    fn new(m: i32) -> Self {
        let n = (m + 1) as usize;
        CP9 {
            m,
            mat: vec![[0.0; ALPHABET_SIZE]; n],
            ins: vec![[0.0; ALPHABET_SIZE]; n],
            t: vec![[0.0; CP9_NTRANS]; n],
            begin: vec![0.0; n],
            end: vec![0.0; n],
            null: [0.0; ALPHABET_SIZE],
            flags: 0,
            has_el: vec![false; n],
            el_self: 0.0,
            el_selfsc: 0,
            el_from_ct: vec![0; n + 1],       // [0..=M+1]
            el_from_idx: vec![Vec::new(); n + 1],
            msc: vec![[0; ALPHABET_SIZE_P]; n],
            isc: vec![[0; ALPHABET_SIZE_P]; n],
            tsc: vec![[0; CP9_NTRANS]; n],
            bsc: {
                // C AllocCPlan9Body: bsc[0] = -INFTY
                let mut v = vec![0; n];
                v[0] = -CP9_INFTY;
                v
            },
            esc: {
                // C AllocCPlan9Body: esc[0] = -INFTY
                let mut v = vec![0; n];
                v[0] = -CP9_INFTY;
                v
            },
            otsc: vec![[0; CP9O_NTRANS]; n],
        }
    }
}

/// C: sreLOG2(x) = (x>0 ? log(x)*1.44269504 : IMPOSSIBLE) (infernal.h:153).
#[inline]
fn sre_log2(x: f64) -> f64 {
    if x > 0.0 {
        x.ln() * 1.44269504
    } else {
        -1e36 // IMPOSSIBLE
    }
}

/// C: sreEXP2(x) = exp(x*0.69314718) (infernal.h:154).
#[inline]
fn sre_exp2(x: f64) -> f64 {
    (x * 0.69314718).exp()
}

/// C: Prob2Score (cm.c:4201). Scaled integer log2-odds, round-to-nearest.
#[inline]
fn prob2score(p: f32, null: f32) -> i32 {
    if p == 0.0 {
        -CP9_INFTY
    } else {
        // sreLOG2(p/null): p/null in f32, then log in f64
        let ratio = p / null;
        (0.5_f64 + 1000.0_f64 * sre_log2(ratio as f64)).floor() as i32
    }
}

/// C: CPlan9Renormalize (cp9.c:571). Normalize all CP9 probability distributions.
pub fn cp9_renormalize(hmm: &mut CP9) {
    let k_abc = ALPHABET_SIZE;
    let mm = hmm.m as usize;
    // match emissions (M_0 is B state, non-emitter)
    for x in hmm.mat[0].iter_mut() {
        *x = 0.0;
    }
    for k in 1..=mm {
        fnorm(&mut hmm.mat[k][..k_abc]);
    }
    // insert emissions
    for k in 0..=mm {
        fnorm(&mut hmm.ins[k][..k_abc]);
    }
    // begin transitions
    let d = fsum(&hmm.begin[1..=mm]) + hmm.t[0][CTMI] + hmm.t[0][CTMD] + hmm.t[0][_CTMEL];
    fscale(&mut hmm.begin[1..=mm], 1.0 / d);
    hmm.t[0][CTMI] /= d;
    hmm.t[0][CTMD] /= d;
    hmm.t[0][_CTMEL] /= d;
    fnorm(&mut hmm.t[0][CP9_TRANS_INSERT_OFFSET..CP9_TRANS_INSERT_OFFSET + 3]);
    for x in hmm.t[0][CP9_TRANS_DELETE_OFFSET..CP9_TRANS_DELETE_OFFSET + 3].iter_mut() {
        *x = 0.0;
    }
    // main model transitions
    for k in 1..=mm {
        let d = fsum(&hmm.t[k][0..4]) + hmm.end[k];
        fscale(&mut hmm.t[k][0..4], 1.0 / d);
        hmm.end[k] /= d;
        fnorm(&mut hmm.t[k][CP9_TRANS_INSERT_OFFSET..CP9_TRANS_INSERT_OFFSET + 3]);
        fnorm(&mut hmm.t[k][CP9_TRANS_DELETE_OFFSET..CP9_TRANS_DELETE_OFFSET + 3]);
    }
    // null model
    fnorm(&mut hmm.null[..k_abc]);

    hmm.flags &= !CPLAN9_HASBITS;
    hmm.flags |= CPLAN9_HASPROB;
}

/// C: CPlan9InitEL (cp9_modelmaker.c:2838) — the score-relevant parts. Sets
/// el_self/el_selfsc and has_el[k] (the el_from_ct/idx DP connectivity is deferred
/// until the DP needs it). Uses the CM's consensus emit map (rpos/lpos).
pub fn cp9_init_el(cm: &CM, emap: &EmitMap, hmm: &mut CP9) {
    // el_self = sreEXP2(cm->el_selfsc); el_selfsc = Prob2Score(el_self, 1.0)
    hmm.el_self = sre_exp2(cm.el_selfsc as f64) as f32;
    hmm.el_selfsc = prob2score(hmm.el_self, 1.0);

    for k in 0..=hmm.m as usize {
        hmm.has_el[k] = false;
    }
    // First pass: has_el + el_from_ct
    for k in 0..hmm.el_from_ct.len() {
        hmm.el_from_ct[k] = 0;
    }
    for nd in 0..cm.nodes as usize {
        let t = cm.ndtype[nd] as i32;
        if (t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BEGL_ND || t == BEGR_ND)
            && cm.ndtype[nd + 1] as i32 != END_ND
        {
            hmm.el_from_ct[emap.rpos[nd] as usize] += 1;
            hmm.has_el[emap.lpos[nd] as usize] = true;
        }
    }
    // Second pass: el_from_idx[k] = source HMM nodes (lpos) for EL into node k=rpos[nd]
    for v in hmm.el_from_idx.iter_mut() {
        v.clear();
    }
    for nd in 0..cm.nodes as usize {
        let t = cm.ndtype[nd] as i32;
        if (t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BEGL_ND || t == BEGR_ND)
            && cm.ndtype[nd + 1] as i32 != END_ND
        {
            let k = emap.rpos[nd] as usize;
            hmm.el_from_idx[k].push(emap.lpos[nd]);
        }
    }
}

/// C: cp9_renormalize_exits (cm_modelconfig.c:784).
fn cp9_renormalize_exits(hmm: &mut CP9) {
    let mm = hmm.m as usize;
    for k in 1..mm {
        let d = fsum(&hmm.t[k][0..4]);
        fscale(&mut hmm.t[k][0..4], (1.0 - hmm.end[k]) / d);
    }
    // node M special: CTMD impossible, CTMM is end[M]
    let d = hmm.t[mm][CTMI] + hmm.t[mm][_CTMEL];
    if (d - 0.0).abs() >= 5e-9 {
        let f = (1.0 - hmm.end[mm]) / d;
        hmm.t[mm][CTMI] *= f;
        hmm.t[mm][_CTMEL] *= f;
    }
    hmm.flags &= !CPLAN9_HASBITS;
}

/// C: cp9_sw_config (cm_modelconfig.c:603) for the default cmsearch call
/// `cp9_sw_config(cp9, pbegin, pbegin, FALSE, ndtype[1])` (do_match_local_cm=FALSE).
pub fn cp9_sw_config(hmm: &mut CP9, pentry: f32, pexit: f32) {
    let mm = hmm.m as usize;
    let s = hmm.t[0][CTMI] + hmm.t[0][CTMD] + hmm.t[0][_CTMEL];
    // begin[1]
    hmm.begin[1] = (1.0 - pentry) * (1.0 - s);
    // begin[2..=M] = FSet
    let val = (pentry * (1.0 - s)) / (mm as f32 - 1.0);
    for k in 2..=mm {
        hmm.begin[k] = val;
    }
    // exit
    let basep = pexit / (mm as f32 - 1.0);
    for k in 1..mm {
        hmm.end[k] = basep / (1.0 - basep * (k as f32 - 1.0));
    }
    cp9_renormalize_exits(hmm);
    hmm.flags &= !CPLAN9_HASBITS;
    hmm.flags |= CPLAN9_LOCAL_BEGIN;
    hmm.flags |= CPLAN9_LOCAL_END;
}

/// C: cp9_EL_local_ends_config (cm_modelconfig.c:683). The CP9 is built on the
/// GLOBAL CM (CMH_LOCAL_END down), so the ELSE branch applies:
/// to_el_prob = cm->pend / nexits.
pub fn cp9_el_local_ends_config(cm: &CM, hmm: &mut CP9) {
    let mut nexits = 0i32;
    for nd in 1..cm.nodes as usize {
        let t = cm.ndtype[nd] as i32;
        if (t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BEGL_ND || t == BEGR_ND)
            && cm.ndtype[nd + 1] as i32 != END_ND
        {
            nexits += 1;
        }
    }
    let to_el_prob = cm.pend / nexits as f32;

    hmm.t[0][_CTMEL] = 0.0;
    for k in 1..=hmm.m as usize {
        if hmm.has_el[k] {
            hmm.t[k][_CTMEL] = to_el_prob;
            let norm_factor = 1.0 - (hmm.t[k][_CTMEL] / (1.0 - hmm.end[k]));
            hmm.t[k][CTMM] *= norm_factor;
            hmm.t[k][CTMI] *= norm_factor;
            hmm.t[k][CTMD] *= norm_factor;
        }
    }
    hmm.flags &= !CPLAN9_HASBITS;
    hmm.flags |= CPLAN9_EL;
}

/// C: `esl_abc_IExpectScore` (esl_alphabet.c). NULL-WEIGHTED average of the
/// canonical integer scores over the degenerate residue `x`, rounded
/// half-away-from-zero. `p` = the CP9 null vector. Non-residue codes return 0
/// (but CP9Logoddsify never calls this for them — they stay -INFTY).
#[inline]
fn cp9_iexpect_score(x: usize, sc: &[i32; ALPHABET_SIZE_P], p: &[f32; ALPHABET_SIZE]) -> i32 {
    if !crate::cm::abc_is_residue(x) {
        return 0;
    }
    let mut result = 0.0f32;
    let mut denom = 0.0f32;
    for i in 0..ALPHABET_SIZE {
        if crate::cm::RNA_DEGEN[x][i] {
            result += sc[i] as f32 * p[i];
            denom += p[i];
        }
    }
    result /= denom;
    if result < 0.0 {
        (result - 0.5) as i32
    } else {
        (result + 0.5) as i32
    }
}

/// C: CP9Logoddsify (cp9.c:463). Integer log-odds scores + reordered otsc.
/// Degenerate Kp emission symbols are filled via esl_abc_IExpectScVec (a
/// null-weighted average of the canonical scores), so a scanned window with
/// IUPAC ambiguity codes (incl. N) marginalizes rather than panics.
pub fn cp9_logoddsify(hmm: &mut CP9) {
    if hmm.flags & CPLAN9_HASBITS != 0 {
        return;
    }
    let mm = hmm.m as usize;
    let k_abc = ALPHABET_SIZE;
    let kp = ALPHABET_SIZE_P;

    // insert emission scores. C: sc[K]=sc[Kp-2]=sc[Kp-1]=-INFTY; canonical x<K
    // from Prob2Score; esl_abc_IExpectScVec fills degenerate x in [K+1 .. Kp-3].
    for k in 0..=mm {
        let mut sc = [0i32; ALPHABET_SIZE_P];
        for x in 0..k_abc {
            sc[x] = prob2score(hmm.ins[k][x], hmm.null[x]);
        }
        sc[k_abc] = -CP9_INFTY; // gap
        sc[kp - 2] = -CP9_INFTY; // nonresidue
        sc[kp - 1] = -CP9_INFTY; // missing
        for x in (k_abc + 1)..=(kp - 3) {
            sc[x] = cp9_iexpect_score(x, &sc, &hmm.null);
        }
        hmm.isc[k] = sc;
    }
    // match emission scores (k=1..M; msc[0] stays all-0)
    for k in 1..=mm {
        let mut sc = [0i32; ALPHABET_SIZE_P];
        for x in 0..k_abc {
            sc[x] = prob2score(hmm.mat[k][x], hmm.null[x]);
        }
        sc[k_abc] = -CP9_INFTY; // gap
        sc[kp - 2] = -CP9_INFTY; // nonresidue
        sc[kp - 1] = -CP9_INFTY; // missing
        for x in (k_abc + 1)..=(kp - 3) {
            sc[x] = cp9_iexpect_score(x, &sc, &hmm.null);
        }
        hmm.msc[k] = sc;
    }
    // transition scores
    for k in 0..=mm {
        hmm.tsc[k][CTMM] = prob2score(hmm.t[k][CTMM], 1.0);
        hmm.tsc[k][CTMI] = prob2score(hmm.t[k][CTMI], 1.0);
        hmm.tsc[k][CTMD] = prob2score(hmm.t[k][CTMD], 1.0);
        hmm.tsc[k][_CTMEL] = prob2score(hmm.t[k][_CTMEL], 1.0);
        hmm.tsc[k][CTIM] = prob2score(hmm.t[k][CTIM], 1.0);
        hmm.tsc[k][CTII] = prob2score(hmm.t[k][CTII], 1.0);
        hmm.tsc[k][CTID] = prob2score(hmm.t[k][CTID], 1.0);
        if k != 0 {
            hmm.tsc[k][CTDM] = prob2score(hmm.t[k][CTDM], 1.0);
            hmm.tsc[k][CTDI] = prob2score(hmm.t[k][CTDI], 1.0);
            hmm.tsc[k][CTDD] = prob2score(hmm.t[k][CTDD], 1.0);
            hmm.bsc[k] = prob2score(hmm.begin[k], 1.0);
            hmm.esc[k] = prob2score(hmm.end[k], 1.0);
        } else {
            hmm.tsc[k][CTDM] = -CP9_INFTY;
            hmm.tsc[k][CTDD] = -CP9_INFTY;
            hmm.tsc[k][CTDI] = -CP9_INFTY;
        }
    }
    hmm.el_selfsc = prob2score(hmm.el_self, 1.0);

    // reordered transition scores
    for k in 0..=mm {
        let o = &mut hmm.otsc[k];
        o[CP9O_MM] = hmm.tsc[k][CTMM];
        o[CP9O_MI] = hmm.tsc[k][CTMI];
        o[CP9O_MD] = hmm.tsc[k][CTMD];
        o[CP9O_IM] = hmm.tsc[k][CTIM];
        o[CP9O_II] = hmm.tsc[k][CTII];
        o[CP9O_DM] = hmm.tsc[k][CTDM];
        o[CP9O_DD] = hmm.tsc[k][CTDD];
        o[CP9O_ID] = hmm.tsc[k][CTID];
        o[CP9O_DI] = hmm.tsc[k][CTDI];
        o[CP9O_BM] = hmm.bsc[k];
        o[CP9O_MEL] = hmm.tsc[k][_CTMEL];
        o[CP9O_ME] = hmm.esc[k];
    }
    hmm.flags |= CPLAN9_HASBITS;
}

/// C: Scorify (infernal.h) = sc / INTSCALE.
#[inline]
fn scorify(sc: i32) -> f32 {
    sc as f32 / 1000.0
}

/// Full CP9 Forward DP matrices (full L+1 rows, be_efficient=FALSE).
pub struct CP9Mx {
    pub l: usize,
    pub m: usize,
    pub mmx: Vec<Vec<i32>>,  // [0..=L][0..=M]
    pub imx: Vec<Vec<i32>>,
    pub dmx: Vec<Vec<i32>>,
    pub elmx: Vec<Vec<i32>>,
    pub erow: Vec<i32>, // [0..=L]
}

/// C: cp9_Forward (cp9_dp.c:666), be_efficient=FALSE (full L+1-row matrix).
/// `dsq[i0..=j0]` are residue codes (0..K-1); dsq is 1-based like Easel's ESL_DSQ.
/// Returns (best_sc, best_pos, matrices, scA).
pub fn cp9_forward(
    hmm: &CP9,
    dsq: &[u8],
    i0: usize,
    j0: usize,
    do_scan: bool,
    doing_align: bool,
) -> (f32, i64, CP9Mx, Vec<i32>) {
    let m = hmm.m as usize;
    let l = j0 - i0 + 1;
    let ninf = -INFTY;
    let mut mmx = vec![vec![ninf; m + 1]; l + 1];
    let mut imx = vec![vec![ninf; m + 1]; l + 1];
    let mut dmx = vec![vec![ninf; m + 1]; l + 1];
    let mut elmx = vec![vec![ninf; m + 1]; l + 1];
    let mut erow = vec![ninf; l + 1];
    let mut sca = vec![0i32; l + 1];

    let mut best_sc: f32 = -1e36; // IMPOSSIBLE
    let mut best_pos: i64 = -1;

    // CP9TSC(x,k) == otsc[k][x]
    macro_rules! tsc {
        ($x:expr, $k:expr) => {
            hmm.otsc[$k][$x]
        };
    }

    // Zero row init
    mmx[0][0] = 0;
    imx[0][0] = ninf;
    dmx[0][0] = ninf;
    elmx[0][0] = ninf;
    erow[0] = ninf;
    for k in 1..=m {
        mmx[0][k] = ninf;
        imx[0][k] = ninf;
        elmx[0][k] = ninf;
        let sc = ilogsum(
            ilogsum(
                mmx[0][k - 1] + tsc!(CP9O_MD, k - 1),
                imx[0][k - 1] + tsc!(CP9O_ID, k - 1),
            ),
            dmx[0][k - 1] + tsc!(CP9O_DD, k - 1),
        );
        dmx[0][k] = sc;
    }
    erow[0] = dmx[0][m] + tsc!(CP9O_DM, m);
    sca[0] = erow[0];
    let fsc = scorify(sca[0]);
    if fsc > best_sc {
        best_sc = fsc;
        best_pos = i0 as i64 - 1;
    }

    // Main loop
    for j in i0..=j0 {
        let dj = dsq[j] as usize;
        debug_assert!(dj < ALPHABET_SIZE_P, "cp9_forward: residue code {} out of alphabet", dj);
        let jp = j - i0 + 1;
        let cur = jp;
        let prv = jp - 1;
        let el_selfsc = hmm.el_selfsc;

        // Hoist the current/previous rows of each matrix once. cur = prv+1 are
        // adjacent, so split_at_mut(cur) yields prv (last of the left half, shared)
        // and cur (first of the right half, mutable) without a Vec<Vec> double-index
        // per cell. (checked slice indexing; byte-identical.)
        let (mmx_lo, mmx_hi) = mmx.split_at_mut(cur);
        let (imx_lo, imx_hi) = imx.split_at_mut(cur);
        let (dmx_lo, dmx_hi) = dmx.split_at_mut(cur);
        let (elmx_lo, elmx_hi) = elmx.split_at_mut(cur);
        let mmx_prv = &mmx_lo[prv];
        let mmx_cur = &mut mmx_hi[0];
        let imx_prv = &imx_lo[prv];
        let imx_cur = &mut imx_hi[0];
        let dmx_prv = &dmx_lo[prv];
        let dmx_cur = &mut dmx_hi[0];
        let elmx_prv = &elmx_lo[prv];
        let elmx_cur = &mut elmx_hi[0];

        mmx_cur[0] = if do_scan { 0 } else { ninf };
        dmx_cur[0] = ninf;
        elmx_cur[0] = ninf;

        let sc = ilogsum(
            ilogsum(
                mmx_prv[0] + tsc!(CP9O_MI, 0),
                imx_prv[0] + tsc!(CP9O_II, 0),
            ),
            dmx_prv[0] + tsc!(CP9O_DI, 0),
        );
        imx_cur[0] = sc + hmm.isc[0][dj];

        let mut endsc = ninf;
        for k in 1..=m {
            // match
            let mut sc = ilogsum(
                ilogsum(
                    mmx_prv[k - 1] + tsc!(CP9O_MM, k - 1),
                    imx_prv[k - 1] + tsc!(CP9O_IM, k - 1),
                ),
                ilogsum(
                    dmx_prv[k - 1] + tsc!(CP9O_DM, k - 1),
                    mmx_prv[0] + tsc!(CP9O_BM, k),
                ),
            );
            for c in 0..hmm.el_from_ct[k] as usize {
                sc = ilogsum(sc, elmx_prv[hmm.el_from_idx[k][c] as usize]);
            }
            mmx_cur[k] = sc + hmm.msc[k][dj];

            // E state update
            endsc = ilogsum(endsc, mmx_cur[k] + tsc!(CP9O_ME, k));

            // insert
            let sc = ilogsum(
                ilogsum(
                    mmx_prv[k] + tsc!(CP9O_MI, k),
                    imx_prv[k] + tsc!(CP9O_II, k),
                ),
                dmx_prv[k] + tsc!(CP9O_DI, k),
            );
            imx_cur[k] = sc + hmm.isc[k][dj];

            // delete
            let sc = ilogsum(
                ilogsum(
                    mmx_cur[k - 1] + tsc!(CP9O_MD, k - 1),
                    imx_cur[k - 1] + tsc!(CP9O_ID, k - 1),
                ),
                dmx_cur[k - 1] + tsc!(CP9O_DD, k - 1),
            );
            dmx_cur[k] = sc;

            // EL
            let mut sc = ninf;
            if (hmm.flags & CPLAN9_EL != 0) && hmm.has_el[k] {
                sc = ilogsum(
                    mmx_cur[k] + tsc!(CP9O_MEL, k),
                    elmx_prv[k] + el_selfsc,
                );
            }
            elmx_cur[k] = sc;
        }
        endsc = ilogsum(
            ilogsum(endsc, dmx_cur[m] + tsc!(CP9O_DM, m)),
            imx_cur[m] + tsc!(CP9O_IM, m),
        );
        for c in 0..hmm.el_from_ct[m + 1] as usize {
            endsc = ilogsum(endsc, elmx_cur[hmm.el_from_idx[m + 1][c] as usize]);
        }
        erow[cur] = endsc;
        sca[jp] = endsc;
        let fsc = scorify(endsc);
        if fsc > best_sc {
            best_sc = fsc;
            best_pos = j as i64;
        }
    }

    if doing_align {
        best_sc = scorify(sca[l]);
        best_pos = i0 as i64;
    }

    let mx = CP9Mx { l, m, mmx, imx, dmx, elmx, erow };
    (best_sc, best_pos, mx, sca)
}

/// CP9 per-state HMM position bands (subset used by cp9_FB2HMMBands).
/// Each array is [0..=M]. Unset states are flagged -1 (new-way, do_old_hmm2ij=FALSE).
#[derive(Clone)]
pub struct CP9Bands {
    pub hmm_m: i32,
    pub pn_min_m: Vec<i32>,
    pub pn_max_m: Vec<i32>,
    pub pn_min_i: Vec<i32>,
    pub pn_max_i: Vec<i32>,
    pub pn_min_d: Vec<i32>,
    pub pn_max_d: Vec<i32>,
    // Filled by cp9_hmm2ijbands: per-CM-state i/j bands. [0..cm.M-1]
    pub imin: Vec<i32>,
    pub imax: Vec<i32>,
    pub jmin: Vec<i32>,
    pub jmax: Vec<i32>,
    // Filled by ij2d_bands: hdmin[v][jp], hdmax[v][jp], jp = j - jmin[v]. [0..cm.M-1]
    pub hdmin: Vec<Vec<i32>>,
    pub hdmax: Vec<Vec<i32>>,
}

/// C: cp9_FB2HMMBands (hmmband.c:577), new-way branch (do_old_hmm2ij=FALSE),
/// use_sums=FALSE. Computes the posterior inline (fmx+bmx-emit-sc) and derives
/// per-state min/max sequence-position bands by creeping in from both ends until
/// `thresh` mass is excluded. `p_thresh = 1 - tau`.
pub fn cp9_fb2hmmbands(
    hmm: &CP9,
    dsq: &[u8],
    fmx: &CP9Mx,
    bmx: &CP9Mx,
    i0: usize,
    j0: usize,
    p_thresh: f64,
    did_fwd_scan: bool,
) -> CP9Bands {
    let m = hmm.m as usize;
    let l = j0 - i0 + 1;
    let ninf = -INFTY;
    // thresh = Prob2Score((1-p_thresh)/2, 1.0)
    let thresh = prob2score(((1.0 - p_thresh) / 2.0) as f32, 1.0);

    let mut pmx_m = vec![vec![ninf; m + 1]; l + 1];
    let mut pmx_i = vec![vec![ninf; m + 1]; l + 1];
    let mut pmx_d = vec![vec![ninf; m + 1]; l + 1];

    let mut mass_m = vec![ninf; m + 1];
    let mut mass_i = vec![ninf; m + 1];
    let mut mass_d = vec![ninf; m + 1];
    let mut nset_m = vec![false; m + 1];
    let mut nset_i = vec![false; m + 1];
    let mut nset_d = vec![false; m + 1];
    let mut xset_m = vec![false; m + 1];
    let mut xset_i = vec![false; m + 1];
    let mut xset_d = vec![false; m + 1];

    let mut b = CP9Bands {
        hmm_m: m as i32,
        pn_min_m: vec![0; m + 1],
        pn_max_m: vec![0; m + 1],
        pn_min_i: vec![0; m + 1],
        pn_max_i: vec![0; m + 1],
        pn_min_d: vec![0; m + 1],
        pn_max_d: vec![0; m + 1],
        imin: Vec::new(),
        imax: Vec::new(),
        jmin: Vec::new(),
        jmax: Vec::new(),
        hdmin: Vec::new(),
        hdmax: Vec::new(),
    };

    // sc = summed log prob of all parses
    let sc = if did_fwd_scan {
        let mut s = ninf;
        for ip in 0..=l {
            s = ilogsum(s, bmx.mmx[ip][0]);
        }
        s
    } else {
        bmx.mmx[0][0]
    };

    let imax0 = |v: i32| if v < 0 { 0 } else { v };

    // Boundary ip=0, i=i0-1
    pmx_m[0][0] = fmx.mmx[0][0] + bmx.mmx[0][0] - sc;
    pmx_i[0][0] = ninf;
    pmx_d[0][0] = ninf;
    mass_m[0] = pmx_m[0][0];
    if mass_m[0] > thresh {
        b.pn_min_m[0] = imax0(i0 as i32 - 1);
        nset_m[0] = true;
    }
    mass_i[0] = ninf;
    mass_d[0] = ninf;
    for k in 1..=m {
        pmx_m[0][k] = ninf;
        pmx_i[0][k] = ninf;
        pmx_d[0][k] = fmx.dmx[0][k] + bmx.dmx[0][k] - sc;
        mass_d[k] = pmx_d[0][k];
        if mass_d[k] > thresh {
            b.pn_min_d[k] = imax0(i0 as i32 - 1);
            nset_d[k] = true;
        }
    }

    // Minimum scan: ip=1..L
    for ip in 1..=l {
        let i = i0 + ip - 1;
        let di = dsq[i] as usize;
        pmx_m[ip][0] = (fmx.mmx[ip][0] + bmx.mmx[ip][0] - sc).max(ninf);
        if !nset_m[0] {
            mass_m[0] = ilogsum(mass_m[0], pmx_m[ip][0]);
            if mass_m[0] > thresh {
                b.pn_min_m[0] = i as i32;
                nset_m[0] = true;
            }
        }
        pmx_i[ip][0] = (fmx.imx[ip][0] + bmx.imx[ip][0] - hmm.isc[0][di] - sc).max(ninf);
        if !nset_i[0] {
            mass_i[0] = ilogsum(mass_i[0], pmx_i[ip][0]);
            if mass_i[0] > thresh {
                b.pn_min_i[0] = i as i32;
                nset_i[0] = true;
            }
        }
        pmx_d[ip][0] = ninf;
        for k in 1..=m {
            pmx_m[ip][k] = (fmx.mmx[ip][k] + bmx.mmx[ip][k] - hmm.msc[k][di] - sc).max(ninf);
            pmx_i[ip][k] = (fmx.imx[ip][k] + bmx.imx[ip][k] - hmm.isc[k][di] - sc).max(ninf);
            pmx_d[ip][k] = (fmx.dmx[ip][k] + bmx.dmx[ip][k] - sc).max(ninf);
            if !nset_m[k] {
                mass_m[k] = ilogsum(mass_m[k], pmx_m[ip][k]);
                if mass_m[k] > thresh {
                    b.pn_min_m[k] = i as i32;
                    nset_m[k] = true;
                }
            }
            if !nset_i[k] {
                mass_i[k] = ilogsum(mass_i[k], pmx_i[ip][k]);
                if mass_i[k] > thresh {
                    b.pn_min_i[k] = i as i32;
                    nset_i[k] = true;
                }
            }
            if !nset_d[k] {
                mass_d[k] = ilogsum(mass_d[k], pmx_d[ip][k]);
                if mass_d[k] > thresh {
                    b.pn_min_d[k] = i as i32;
                    nset_d[k] = true;
                }
            }
        }
    }

    // Maximum scan: ip=L..1
    for v in mass_m.iter_mut() {
        *v = ninf;
    }
    for v in mass_i.iter_mut() {
        *v = ninf;
    }
    for v in mass_d.iter_mut() {
        *v = ninf;
    }
    for ip in (1..=l).rev() {
        let i = i0 + ip - 1;
        for k in 0..=m {
            if !xset_m[k] {
                mass_m[k] = ilogsum(mass_m[k], pmx_m[ip][k]);
                if mass_m[k] > thresh {
                    b.pn_max_m[k] = i as i32;
                    xset_m[k] = true;
                }
            }
            if !xset_i[k] {
                mass_i[k] = ilogsum(mass_i[k], pmx_i[ip][k]);
                if mass_i[k] > thresh {
                    b.pn_max_i[k] = i as i32;
                    xset_i[k] = true;
                }
            }
            if !xset_d[k] {
                mass_d[k] = ilogsum(mass_d[k], pmx_d[ip][k]);
                if mass_d[k] > thresh {
                    b.pn_max_d[k] = i as i32;
                    xset_d[k] = true;
                }
            }
        }
    }
    // Boundary ip=0
    if !xset_m[0] {
        mass_m[0] = ilogsum(mass_m[0], pmx_m[0][0]);
        if mass_m[0] > thresh {
            b.pn_max_m[0] = imax0(i0 as i32 - 1);
            xset_m[0] = true;
        }
    }
    for k in 1..=m {
        if !xset_d[k] {
            mass_d[k] = ilogsum(mass_d[k], pmx_d[0][k]);
            if mass_d[k] > thresh {
                b.pn_max_d[k] = imax0(i0 as i32 - 1);
                xset_d[k] = true;
            }
        }
    }

    // New-way: flag unset (or inverted) bands with -1
    for k in 0..=m {
        if !nset_m[k] || !xset_m[k] || b.pn_max_m[k] < b.pn_min_m[k] {
            b.pn_min_m[k] = -1;
            b.pn_max_m[k] = -1;
        }
        if !nset_i[k] || !xset_i[k] || b.pn_max_i[k] < b.pn_min_i[k] {
            b.pn_min_i[k] = -1;
            b.pn_max_i[k] = -1;
        }
        if !nset_d[k] || !xset_d[k] || b.pn_max_d[k] < b.pn_min_d[k] {
            b.pn_min_d[k] = -1;
            b.pn_max_d[k] = -1;
        }
    }
    b.pn_min_d[0] = -1; // D_0 doesn't exist
    b.pn_max_d[0] = -1;
    b
}

/// C: cp9_Backward (cp9_dp.c:992), be_efficient=FALSE (full L+1-row matrix).
/// Returns (best_sc, best_pos, matrices, scA[0..=L]).
pub fn cp9_backward(
    hmm: &CP9,
    dsq: &[u8],
    i0: usize,
    j0: usize,
    do_scan: bool,
    doing_align: bool,
) -> (f32, i64, CP9Mx, Vec<i32>) {
    let m = hmm.m as usize;
    let l = j0 - i0 + 1;
    let ninf = -INFTY;
    let el = hmm.flags & CPLAN9_EL != 0;
    let el_selfsc = hmm.el_selfsc;
    let mut mmx = vec![vec![ninf; m + 1]; l + 1];
    let mut imx = vec![vec![ninf; m + 1]; l + 1];
    let mut dmx = vec![vec![ninf; m + 1]; l + 1];
    let mut elmx = vec![vec![ninf; m + 1]; l + 1];
    let erow = vec![ninf; l + 1];
    let mut sca = vec![0i32; l + 2];

    macro_rules! tsc {
        ($x:expr, $k:expr) => {
            hmm.otsc[$k][$x]
        };
    }
    let mut best_sc: f32 = -1e36;
    let mut best_pos: i64 = -1;

    // --- Initialization row: i=j0, cur=L ---
    let cur = l;
    let dj0 = dsq[j0] as usize;
    for k in 1..=m {
        elmx[cur][k] = ninf;
    }
    if el {
        for c in 0..hmm.el_from_ct[m + 1] as usize {
            elmx[cur][hmm.el_from_idx[m + 1][c] as usize] = 0;
        }
    }
    mmx[cur][m] = ilogsum(elmx[cur][m] + tsc!(CP9O_MEL, m), tsc!(CP9O_ME, m)) + hmm.msc[m][dj0];
    imx[cur][m] = tsc!(CP9O_IM, m) + hmm.isc[m][dj0];
    dmx[cur][m] = tsc!(CP9O_DM, m);
    for k in (1..m).rev() {
        let mut v = tsc!(CP9O_ME, k);
        v = ilogsum(v, dmx[cur][k + 1] + tsc!(CP9O_MD, k));
        if el {
            v = ilogsum(v, elmx[cur][k] + tsc!(CP9O_MEL, k));
        }
        mmx[cur][k] = v + hmm.msc[k][dj0];
        imx[cur][k] = dmx[cur][k + 1] + tsc!(CP9O_ID, k) + hmm.isc[k][dj0];
        dmx[cur][k] = dmx[cur][k + 1] + tsc!(CP9O_DD, k);
    }
    mmx[cur][0] = dmx[cur][1] + tsc!(CP9O_MD, 0);
    imx[cur][0] = dmx[cur][1] + tsc!(CP9O_ID, 0) + hmm.isc[0][dj0];
    dmx[cur][0] = ninf;
    elmx[cur][0] = ninf;
    sca[l] = mmx[cur][0]; // C does NOT update best_sc on the init row

    // --- Main loop: i = j0-1 down to i0 ---
    for i in (i0..j0).rev() {
        let ip = i - i0 + 1;
        let cur = ip;
        let prv = ip + 1;
        let di = dsq[i] as usize;

        // Hoist current/previous rows. Here prv = cur+1 (backward pass); dmx is
        // read/written on the cur row only, so it needs just a single mutable row
        // ref, while mmx/imx/elmx also read the prv row → split_at_mut(prv).
        // (checked slice indexing; byte-identical.)
        let (mmx_lo, mmx_hi) = mmx.split_at_mut(prv);
        let (imx_lo, imx_hi) = imx.split_at_mut(prv);
        let (elmx_lo, elmx_hi) = elmx.split_at_mut(prv);
        let mmx_cur = &mut mmx_lo[cur];
        let mmx_prv = &mmx_hi[0];
        let imx_cur = &mut imx_lo[cur];
        let imx_prv = &imx_hi[0];
        let elmx_cur = &mut elmx_lo[cur];
        let elmx_prv = &elmx_hi[0];
        let dmx_cur = &mut dmx[cur];

        for k in 0..=m {
            elmx_cur[k] = ninf;
        }
        if el && hmm.has_el[m] {
            elmx_cur[m] += el_selfsc; // faithful: -INFTY + el_selfsc
        }
        mmx_cur[m] = imx_prv[m] + tsc!(CP9O_MI, m) + hmm.msc[m][di];
        if el && hmm.has_el[m] {
            mmx_cur[m] = ilogsum(mmx_cur[m], elmx_cur[m] + tsc!(CP9O_MEL, m));
        }
        imx_cur[m] = imx_prv[m] + tsc!(CP9O_II, m) + hmm.isc[m][di];
        dmx_cur[m] = imx_prv[m] + tsc!(CP9O_DI, m);
        if el {
            for c in 0..hmm.el_from_ct[m] as usize {
                let src = hmm.el_from_idx[m][c] as usize;
                elmx_cur[src] = ilogsum(elmx_cur[src], mmx_prv[m]);
            }
        }
        if do_scan {
            if el {
                for c in 0..hmm.el_from_ct[m + 1] as usize {
                    elmx_cur[hmm.el_from_idx[m + 1][c] as usize] = 0;
                }
            }
            mmx_cur[m] = ilogsum(
                mmx_cur[m],
                ilogsum(elmx_cur[m] + tsc!(CP9O_MEL, m), tsc!(CP9O_ME, m)),
            );
            imx_cur[m] = ilogsum(imx_cur[m], tsc!(CP9O_IM, m));
            dmx_cur[m] = ilogsum(dmx_cur[m], tsc!(CP9O_DM, m));
        }
        for k in (1..m).rev() {
            if el {
                for c in 0..hmm.el_from_ct[k] as usize {
                    let src = hmm.el_from_idx[k][c] as usize;
                    elmx_cur[src] = ilogsum(elmx_cur[src], mmx_prv[k]);
                }
            }
            if el && hmm.has_el[k] {
                elmx_cur[k] = ilogsum(elmx_cur[k], elmx_prv[k] + el_selfsc);
            }
            let mut mv = ilogsum(
                ilogsum(mmx_prv[k + 1] + tsc!(CP9O_MM, k), imx_prv[k] + tsc!(CP9O_MI, k)),
                dmx_cur[k + 1] + tsc!(CP9O_MD, k),
            );
            if el && hmm.has_el[k] {
                mv = ilogsum(mv, elmx_cur[k] + tsc!(CP9O_MEL, k));
            }
            mmx_cur[k] = mv + hmm.msc[k][di];
            imx_cur[k] = ilogsum(
                ilogsum(mmx_prv[k + 1] + tsc!(CP9O_IM, k), imx_prv[k] + tsc!(CP9O_II, k)),
                dmx_cur[k + 1] + tsc!(CP9O_ID, k),
            ) + hmm.isc[k][di];
            if do_scan {
                mmx_cur[k] = ilogsum(mmx_cur[k], tsc!(CP9O_ME, k));
            }
            dmx_cur[k] = ilogsum(
                ilogsum(mmx_prv[k + 1] + tsc!(CP9O_DM, k), imx_prv[k] + tsc!(CP9O_DI, k)),
                dmx_cur[k + 1] + tsc!(CP9O_DD, k),
            );
        }
        // k == 0
        imx_cur[0] = ilogsum(
            ilogsum(mmx_prv[1] + tsc!(CP9O_IM, 0), imx_prv[0] + tsc!(CP9O_II, 0)),
            dmx_cur[1] + tsc!(CP9O_ID, 0),
        ) + hmm.isc[0][di];
        dmx_cur[0] = ninf;
        elmx_cur[0] = ninf;
        let mut b = ninf;
        for k in (1..=m).rev() {
            b = ilogsum(b, mmx_prv[k] + tsc!(CP9O_BM, k));
        }
        b = ilogsum(b, imx_prv[0] + tsc!(CP9O_MI, 0));
        b = ilogsum(b, dmx_cur[1] + tsc!(CP9O_MD, 0));
        mmx_cur[0] = b;
        sca[ip] = mmx_cur[0];
        let fsc = scorify(sca[ip]);
        if fsc > best_sc {
            best_sc = fsc;
            best_pos = i as i64 + 1;
        }
    }

    // --- Special case ip == 0, i = i0-1, cur=0, prv=1 ---
    {
        let cur = 0usize;
        let prv = 1usize;
        for k in 1..=m {
            elmx[cur][k] = ninf;
        }
        mmx[cur][m] = ninf;
        imx[cur][m] = ninf;
        elmx[cur][m] = ninf;
        dmx[cur][m] = imx[prv][m] + tsc!(CP9O_DI, m);
        if do_scan {
            dmx[cur][m] = ilogsum(dmx[cur][m], tsc!(CP9O_DM, m));
        }
        for k in (1..m).rev() {
            mmx[cur][k] = ninf;
            imx[cur][k] = ninf;
            elmx[cur][k] = ninf;
            dmx[cur][k] = ilogsum(
                ilogsum(mmx[prv][k + 1] + tsc!(CP9O_DM, k), imx[prv][k] + tsc!(CP9O_DI, k)),
                dmx[cur][k + 1] + tsc!(CP9O_DD, k),
            );
        }
        imx[cur][0] = ninf;
        dmx[cur][0] = ninf;
        elmx[cur][0] = ninf;
        let mut b = ninf;
        for k in (1..=m).rev() {
            b = ilogsum(b, mmx[prv][k] + tsc!(CP9O_BM, k));
        }
        b = ilogsum(b, imx[prv][0] + tsc!(CP9O_MI, 0));
        b = ilogsum(b, dmx[cur][1] + tsc!(CP9O_MD, 0));
        mmx[cur][0] = b;
        sca[0] = mmx[cur][0];
        let fsc = scorify(sca[0]);
        if fsc > best_sc {
            best_sc = fsc;
            best_pos = i0 as i64; // i = i0-1, i+1 = i0
        }
    }

    if doing_align {
        best_sc = scorify(sca[0]);
        best_pos = i0 as i64;
    }
    let mx = CP9Mx { l, m, mmx, imx, dmx, elmx, erow };
    (best_sc, best_pos, mx, sca)
}

/// Full CP9 config for the default cmsearch STD pass: build (global) → InitEL →
/// Renormalize → local sw_config → EL config → logoddsify. Matches the C order in
/// cm_Configure (build_cp9_hmm then cm_modelconfig.c:312-354).
pub fn cp9_build_and_configure(cm: &CM, emap: &EmitMap, map: &CP9Map, psi: &[f64], tmap: &Tmap) -> CP9 {
    let mut hmm = cp9_build_emissions(cm, map, psi);
    cp9_init_el(cm, emap, &mut hmm);
    cp9_build_transitions(cm, &mut hmm, map, psi, tmap);
    cp9_renormalize(&mut hmm);
    cp9_sw_config(&mut hmm, cm.pbegin, cm.pbegin);
    cp9_el_local_ends_config(cm, &mut hmm);
    cp9_logoddsify(&mut hmm);
    hmm
}

/// Full CP9 config for a GLOBAL (glocal, `-g`) cmsearch pass: build (global) →
/// InitEL → transitions → Renormalize → logoddsify. Mirrors C `build_cp9_hmm`
/// followed by `CP9Logoddsify` with NO `cp9_sw_config` and NO EL config (which
/// C only applies when CM_CONFIG_HMMLOCAL is set). Used by the `-g --mid` path.
pub fn cp9_build_and_configure_global(
    cm: &CM,
    emap: &EmitMap,
    map: &CP9Map,
    psi: &[f64],
    tmap: &Tmap,
) -> CP9 {
    let mut hmm = cp9_build_emissions(cm, map, psi);
    cp9_init_el(cm, emap, &mut hmm);
    cp9_build_transitions(cm, &mut hmm, map, psi, tmap);
    cp9_renormalize(&mut hmm);
    cp9_logoddsify(&mut hmm);
    hmm
}

/// C: esl_vec_FSum (Kahan) over a slice.
#[inline]
fn fsum(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for &vi in v {
        let y = vi - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// C: esl_vec_FScale.
#[inline]
fn fscale(v: &mut [f32], s: f32) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// C: esl_vec_FNorm.
#[inline]
fn fnorm(v: &mut [f32]) {
    let sum = fsum(v);
    let n = v.len();
    if sum != 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    } else {
        for x in v.iter_mut() {
            *x = 1.0 / n as f32;
        }
    }
}

/// C: cm2hmm_emit_prob (cp9_modelmaker.c:798). Probability CM state x emits
/// residue i, marginalizing the correct half of a MATP_MP pair.
fn cm2hmm_emit_prob(cm: &CM, map: &CP9Map, x: usize, i: usize, k: usize) -> f32 {
    let k_abc = ALPHABET_SIZE;
    let is_left = map.nd2lpos[map.pos2nd[k] as usize] == k as i32;
    if cm.stid[x] as i32 != MATP_MP {
        cm.e[x][i]
    } else {
        let mut ret = 0.0f32;
        if is_left {
            // C: for j=i*K..(i+1)*K-1: ret += e[x][j]
            for j in (i * k_abc)..((i + 1) * k_abc) {
                ret += cm.e[x][j];
            }
        } else {
            // C: for j=i; j<K*K; j+=K: ret += e[x][j]
            let mut j = i;
            while j < k_abc * k_abc {
                ret += cm.e[x][j];
                j += k_abc;
            }
        }
        ret
    }
}

/// C: build_cp9_hmm emission block (cp9_modelmaker.c:262-307). Fills raw
/// (pre-CPlan9Renormalize) match/insert emissions weighted by psi occupancy.
pub fn cp9_build_emissions(cm: &CM, map: &CP9Map, psi: &[f64]) -> CP9 {
    let mm = map.hmm_m;
    let k_abc = ALPHABET_SIZE;
    let mut hmm = CP9::new(mm);

    // C: CPlan9SetNullModel(hmm, cm->null, 1.0) (cp9_modelmaker.c:221)
    for i in 0..k_abc {
        hmm.null[i] = cm.null[i];
    }

    // Special case: 1st insert state maps to CM state 1 (cp9_modelmaker.c:262)
    for i in 0..k_abc {
        hmm.ins[0][i] = cm.e[1][i];
    }

    for k in 1..=mm as usize {
        // Match state (cp9_modelmaker.c:275)
        let a0 = map.hns2cs[k][HMMMATCH][0];
        let a1 = map.hns2cs[k][HMMMATCH][1];
        for i in 0..k_abc {
            // C: mat[k][i] += psi[a0] * emit_prob (double math, stored as f32)
            let cur = hmm.mat[k][i] as f64;
            hmm.mat[k][i] = (cur + psi[a0 as usize] * cm2hmm_emit_prob(cm, map, a0 as usize, i, k) as f64) as f32;
            if a1 != -1 {
                let cur = hmm.mat[k][i] as f64;
                hmm.mat[k][i] =
                    (cur + psi[a1 as usize] * cm2hmm_emit_prob(cm, map, a1 as usize, i, k) as f64) as f32;
            }
        }

        // Insert state (cp9_modelmaker.c:293)
        let b0 = map.hns2cs[k][HMMINSERT][0];
        let b1 = map.hns2cs[k][HMMINSERT][1];
        for i in 0..k_abc {
            let cur = hmm.ins[k][i] as f64;
            hmm.ins[k][i] = (cur + psi[b0 as usize] * cm2hmm_emit_prob(cm, map, b0 as usize, i, k) as f64) as f32;
            if b1 != -1 {
                let cur = hmm.ins[k][i] as f64;
                hmm.ins[k][i] =
                    (cur + psi[b1 as usize] * cm2hmm_emit_prob(cm, map, b1 as usize, i, k) as f64) as f32;
            }
        }
    }
    hmm
}

// =============================================================================
// (ii-e) CP9 transitions — cm2hmm_special_trans_cp9 + cm2hmm_trans_probs_cp9
// (cp9_modelmaker.c:848, 1240) via hmm_add_single_trans_cp9 (:1591) +
// cm_sum_subpaths_cp9 (:1642).
// =============================================================================

type Tmap = Vec<Vec<Vec<i8>>>;

/// C: cm_sum_subpaths_cp9 (cp9_modelmaker.c:1642). Summed probability of all CM
/// subpaths from `start` to `end`, ignoring the start/end self-loops and the
/// insert->start contribution that maps to a different HMM transition.
fn cm_sum_subpaths_cp9(
    cm: &CM,
    map: &CP9Map,
    start: usize,
    end: usize,
    tmap: &Tmap,
    k: usize,
    psi: &[f64],
) -> f32 {
    if start > end {
        panic!("cm_sum_subpaths_cp9: start {} > end {}", start, end);
    }
    if start == end {
        if cm.sttype[start] as i32 != IL_ST && cm.sttype[start] as i32 != IR_ST {
            // must be MATP_MP/MATP_D/MATP_ML/MATP_MR; contribution is 1.0
            return 1.0;
        }
        // self-insert probability
        return cm.t[start][0];
    }
    let nlen = end - start + 1;
    let mut sub_psi = vec![0.0f64; nlen];
    sub_psi[0] = 1.0; // must start in "start"

    if (cm.sttype[start] as i32 != IL_ST && cm.sttype[start] as i32 != IR_ST)
        && (cm.sttype[end] as i32 != IL_ST && cm.sttype[end] as i32 != IR_ST)
    {
        let mut insert_to_start = 0.0f64;
        let h0 = map.hns2cs[k][1][0];
        if h0 < start as i32 {
            insert_to_start =
                psi[h0 as usize] * cm_sum_subpaths_cp9(cm, map, h0 as usize, start, tmap, k, psi) as f64;
        }
        let h1 = map.hns2cs[k][1][1];
        if h1 != -1 && h1 < start as i32 {
            insert_to_start +=
                psi[h1 as usize] * cm_sum_subpaths_cp9(cm, map, h1 as usize, start, tmap, k, psi) as f64;
        }
        sub_psi[0] -= insert_to_start / psi[start];
    }

    for v in (start + 1)..=end {
        let vi = v - start;
        let is_insert = if cm.sttype[v] as i32 == IL_ST || cm.sttype[v] as i32 == IR_ST {
            1
        } else {
            0
        };
        if cm.sttype[v] as i32 == S_ST {
            // prev is BIF_B or END_E; treat transition INTO this S as prob 1.0
            sub_psi[vi] = sub_psi[vi - 1] * 1.0;
        }
        if v != end
            && is_insert == 1
            && (map.cs2hn[v][0] == k as i32 || map.cs2hn[v][1] == k as i32)
        {
            // skip: this insert maps to node k, counted elsewhere
        } else {
            let mut y = cm.pnum[v] - 1;
            while y >= is_insert {
                let x = cm.plast[v] - y; // parent of v
                let tmap_val = tmap[cm.stid[x as usize] as usize]
                    [cm.ndtype[(cm.ndidx[v] + is_insert) as usize] as usize]
                    [cm.stid[v] as usize];
                if (x - start as i32) >= 0 {
                    sub_psi[vi] += sub_psi[(x - start as i32) as usize] * cm.t[x as usize][tmap_val as usize] as f64;
                }
                y -= 1;
            }
            if v != end && is_insert == 1 {
                // contribution of the self insertion loops
                sub_psi[vi] += sub_psi[vi] * (cm.t[v][0] as f64 / (1.0 - cm.t[v][0] as f64));
            }
        }
    }
    sub_psi[end - start] as f32
}

/// C: hmm_add_single_trans_cp9 (cp9_modelmaker.c:1591). Add a virtual-count
/// contribution to a single CP9 transition.
fn hmm_add_single_trans_cp9(
    cm: &CM,
    hmm: &mut CP9,
    map: &CP9Map,
    a: i32,
    b: i32,
    k: usize,
    idx: usize,
    tmap: &Tmap,
    psi: &[f64],
) {
    if a == -1 || b == -1 {
        return;
    }
    if a <= b {
        // going DOWN the CM
        let sub = cm_sum_subpaths_cp9(cm, map, a as usize, b as usize, tmap, k, psi);
        hmm.t[k][idx] = (hmm.t[k][idx] as f64 + psi[a as usize] * sub as f64) as f32;
    } else {
        // going UP the CM
        let sub = cm_sum_subpaths_cp9(cm, map, b as usize, a as usize, tmap, k, psi);
        hmm.t[k][idx] = (hmm.t[k][idx] as f64 + psi[b as usize] * sub as f64) as f32;
    }
}

/// The 4-way (ap[0..1] × bp[0..1]) add pattern used throughout the C transition
/// code.
#[allow(clippy::too_many_arguments)]
fn add4(
    cm: &CM,
    hmm: &mut CP9,
    map: &CP9Map,
    ap: [i32; 2],
    bp: [i32; 2],
    k: usize,
    idx: usize,
    tmap: &Tmap,
    psi: &[f64],
) {
    hmm_add_single_trans_cp9(cm, hmm, map, ap[0], bp[0], k, idx, tmap, psi);
    hmm_add_single_trans_cp9(cm, hmm, map, ap[0], bp[1], k, idx, tmap, psi);
    hmm_add_single_trans_cp9(cm, hmm, map, ap[1], bp[0], k, idx, tmap, psi);
    hmm_add_single_trans_cp9(cm, hmm, map, ap[1], bp[1], k, idx, tmap, psi);
}

/// C: cm2hmm_special_trans_cp9 (cp9_modelmaker.c:848). Transitions INTO node 1
/// (+ N->N) and OUT of node M, plus begin[1]/end[M].
fn cm2hmm_special_trans_cp9(cm: &CM, hmm: &mut CP9, map: &CP9Map, psi: &[f64], tmap: &Tmap) {
    let mm = hmm.m as usize;

    // --- into node 1 ---
    // T1 CTMM: B -> M_1 (becomes begin[1])
    add4(cm, hmm, map, map.hns2cs[0][HMMMATCH], map.hns2cs[1][HMMMATCH], 0, CTMM, tmap, psi);
    hmm.begin[1] = hmm.t[0][CTMM];
    hmm.t[0][CTMM] = 0.0;
    // T2 CTMI: B -> N
    add4(cm, hmm, map, map.hns2cs[0][HMMMATCH], map.hns2cs[0][HMMINSERT], 0, CTMI, tmap, psi);
    // T3 CTMD: B -> D_1
    add4(cm, hmm, map, map.hns2cs[0][HMMMATCH], map.hns2cs[1][HMMDELETE], 0, CTMD, tmap, psi);
    // T4 CTIM: N -> M_1
    add4(cm, hmm, map, map.hns2cs[0][HMMINSERT], map.hns2cs[1][HMMMATCH], 0, CTIM, tmap, psi);
    // T5 CTII: N -> N
    add4(cm, hmm, map, map.hns2cs[0][HMMINSERT], map.hns2cs[0][HMMINSERT], 0, CTII, tmap, psi);
    // T6 CTID: N -> D_1
    add4(cm, hmm, map, map.hns2cs[0][HMMINSERT], map.hns2cs[1][HMMDELETE], 0, CTID, tmap, psi);
    // T7-9: no D_0 state
    hmm.t[0][CTDM] = 0.0;
    hmm.t[0][CTDI] = 0.0;
    hmm.t[0][CTDD] = 0.0;
    // normalize node 0
    let d = fsum(&hmm.begin[1..=mm]) + hmm.t[0][CTMI] + hmm.t[0][CTMD];
    fscale(&mut hmm.begin[1..=mm], 1.0 / d);
    hmm.t[0][CTMI] /= d;
    hmm.t[0][CTMD] /= d;
    fnorm(&mut hmm.t[0][CP9_TRANS_INSERT_OFFSET..CP9_TRANS_INSERT_OFFSET + 3]);

    // --- out of node M ---
    let end_e = [cm.m - 1, -1];
    // T1 CTMM: M_M -> E (becomes end[M])
    add4(cm, hmm, map, map.hns2cs[mm][HMMMATCH], end_e, mm, CTMM, tmap, psi);
    hmm.end[mm] = hmm.t[mm][CTMM];
    hmm.t[mm][CTMM] = 0.0;
    // T2 CTMI: M_M -> I_M
    add4(cm, hmm, map, map.hns2cs[mm][HMMMATCH], map.hns2cs[mm][HMMINSERT], mm, CTMI, tmap, psi);
    // T3 CTMD illegal
    hmm.t[mm][CTMD] = 0.0;
    // T4 CTIM: I_M -> E
    add4(cm, hmm, map, map.hns2cs[mm][HMMINSERT], end_e, mm, CTIM, tmap, psi);
    // T5 CTII: I_M -> I_M
    add4(cm, hmm, map, map.hns2cs[mm][HMMINSERT], map.hns2cs[mm][HMMINSERT], mm, CTII, tmap, psi);
    // T6 CTID illegal
    hmm.t[mm][CTID] = 0.0;
    // T7 CTDM: D_M -> E
    add4(cm, hmm, map, map.hns2cs[mm][HMMDELETE], end_e, mm, CTDM, tmap, psi);
    // T8 CTDI: D_M -> I_M
    add4(cm, hmm, map, map.hns2cs[mm][HMMDELETE], map.hns2cs[mm][HMMINSERT], mm, CTDI, tmap, psi);
    // T9 CTDD illegal
    hmm.t[mm][CTDD] = 0.0;
    // normalize node M
    let d = fsum(&hmm.t[mm][0..4]) + hmm.end[mm];
    fscale(&mut hmm.t[mm][0..4], 1.0 / d);
    hmm.end[mm] /= d;
    fnorm(&mut hmm.t[mm][CP9_TRANS_INSERT_OFFSET..CP9_TRANS_INSERT_OFFSET + 3]);
    fnorm(&mut hmm.t[mm][CP9_TRANS_DELETE_OFFSET..CP9_TRANS_DELETE_OFFSET + 3]);
}

/// C: cm2hmm_trans_probs_cp9 (cp9_modelmaker.c:1240). All 9 transitions out of
/// interior HMM node k (1..M-1).
fn cm2hmm_trans_probs_cp9(cm: &CM, hmm: &mut CP9, map: &CP9Map, k: usize, psi: &[f64], tmap: &Tmap) {
    let hm = map.hns2cs[k][HMMMATCH];
    let hi = map.hns2cs[k][HMMINSERT];
    let hd = map.hns2cs[k][HMMDELETE];
    let nm = map.hns2cs[k + 1][HMMMATCH];
    let nd = map.hns2cs[k + 1][HMMDELETE];
    add4(cm, hmm, map, hm, nm, k, CTMM, tmap, psi); // T1 M_k -> M_k+1
    add4(cm, hmm, map, hm, hi, k, CTMI, tmap, psi); // T2 M_k -> I_k
    add4(cm, hmm, map, hm, nd, k, CTMD, tmap, psi); // T3 M_k -> D_k+1
    add4(cm, hmm, map, hi, nm, k, CTIM, tmap, psi); // T4 I_k -> M_k+1
    add4(cm, hmm, map, hi, hi, k, CTII, tmap, psi); // T5 I_k -> I_k
    add4(cm, hmm, map, hi, nd, k, CTID, tmap, psi); // T6 I_k -> D_k+1
    add4(cm, hmm, map, hd, nm, k, CTDM, tmap, psi); // T7 D_k -> M_k+1
    add4(cm, hmm, map, hd, hi, k, CTDI, tmap, psi); // T8 D_k -> I_k
    add4(cm, hmm, map, hd, nd, k, CTDD, tmap, psi); // T9 D_k -> D_k+1
    // normalize
    let d = fsum(&hmm.t[k][0..4]) + hmm.end[k];
    fscale(&mut hmm.t[k][0..4], 1.0 / d);
    hmm.end[k] /= d;
    fnorm(&mut hmm.t[k][CP9_TRANS_INSERT_OFFSET..CP9_TRANS_INSERT_OFFSET + 3]);
    fnorm(&mut hmm.t[k][CP9_TRANS_DELETE_OFFSET..CP9_TRANS_DELETE_OFFSET + 3]);
}

/// C: build_cp9_hmm transition block (cp9_modelmaker.c:325-330). Fills all CP9
/// transitions (special + per interior node), pre-CPlan9Renormalize.
pub fn cp9_build_transitions(cm: &CM, hmm: &mut CP9, map: &CP9Map, psi: &[f64], tmap: &Tmap) {
    cm2hmm_special_trans_cp9(cm, hmm, map, psi, tmap);
    for k in 1..hmm.m as usize {
        cm2hmm_trans_probs_cp9(cm, hmm, map, k, psi, tmap);
    }
    let _ = HMMDELETE;
}

// ===========================================================================
// (vi) cp9_HMM2ijBands: convert HMM position bands (pn_*) into CM per-state
//      i/j bands (imin/imax/jmin/jmax). Faithful transcription of hmmband.c.
// ===========================================================================

const CMH_LOCAL_BEGIN: u32 = 1 << 10; // infernal.h:1937

/// C: StateDelta (cm.c:1177). # residues emitted by a state.
fn state_delta(sttype: i32) -> i32 {
    match sttype {
        crate::constants::D_ST => 0,
        crate::constants::MP_ST => 2,
        crate::constants::ML_ST => 1,
        crate::constants::MR_ST => 1,
        IL_ST => 1,
        IR_ST => 1,
        S_ST => 0,
        E_ST => 0,
        crate::constants::B_ST => 0,
        EL_ST => 0,
        _ => panic!("bogus state type {}", sttype),
    }
}

/// C: StateLeftDelta (cm.c:1196). # residues emitted to the left by a state.
fn state_left_delta(sttype: i32) -> i32 {
    match sttype {
        crate::constants::D_ST => 0,
        crate::constants::MP_ST => 1,
        crate::constants::ML_ST => 1,
        crate::constants::MR_ST => 0,
        IL_ST => 1,
        IR_ST => 0,
        S_ST => 0,
        E_ST => 0,
        crate::constants::B_ST => 0,
        EL_ST => 0,
        _ => panic!("bogus state type {}", sttype),
    }
}

/// The 10 "reachable residue" arrays filled by HMMBandsEnforceValidParse.
pub struct RArrays {
    pub r_mn: Vec<i32>,
    pub r_mx: Vec<i32>,
    pub r_in: Vec<i32>,
    pub r_ix: Vec<i32>,
    pub r_dn: Vec<i32>,
    pub r_dx: Vec<i32>,
    pub r_nn_i: Vec<i32>,
    pub r_nx_i: Vec<i32>,
    pub r_nn_j: Vec<i32>,
    pub r_nx_j: Vec<i32>,
}

/// C: HMMBandsFixUnreachable (hmmband.c:2899). Expand HMM bands so a parse
/// becomes possible up through node k. Only ever called with local off.
fn hmm_bands_fix_unreachable(cp9b: &mut CP9Bands, k: usize, r_prv_min: i32, _r_prv_max: i32, _r_insert_prv_min: i32) {
    let mut nxt_m: i32 = -1;
    let mut nxt_d: i32 = -1;
    if cp9b.pn_min_m[k] != -1 && cp9b.pn_max_m[k] != -1 {
        if cp9b.pn_max_m[k] - 1 > r_prv_min {
            nxt_m = (cp9b.pn_min_m[k] - 1).max(r_prv_min);
        }
    }
    if cp9b.pn_min_d[k] != -1 && cp9b.pn_max_d[k] != -1 {
        if cp9b.pn_max_d[k] > r_prv_min {
            nxt_d = cp9b.pn_min_d[k].max(r_prv_min);
        }
    }
    if nxt_m != -1 || nxt_d != -1 {
        /* scenario 1 */
        let nxt_n = if nxt_m == -1 {
            nxt_d
        } else if nxt_d == -1 {
            nxt_m
        } else {
            nxt_m.min(nxt_d)
        };
        if cp9b.pn_min_i[k - 1] != -1 {
            cp9b.pn_min_i[k - 1] = cp9b.pn_min_i[k - 1].min(r_prv_min + 1);
        } else {
            cp9b.pn_min_i[k - 1] = r_prv_min + 1;
        }
        if cp9b.pn_max_i[k - 1] != -1 {
            cp9b.pn_max_i[k - 1] = cp9b.pn_max_i[k - 1].max(nxt_n);
        } else {
            cp9b.pn_max_i[k - 1] = nxt_n;
        }
    } else {
        /* scenario 2 */
        let mut kp = k;
        while kp <= cp9b.hmm_m as usize
            && (cp9b.pn_max_m[kp] < (r_prv_min + 1)) && (cp9b.pn_max_d[kp] < r_prv_min)
        {
            cp9b.pn_min_d[kp] = r_prv_min;
            cp9b.pn_max_d[kp] = r_prv_min;
            kp += 1;
        }
    }
}

/// C: HMMBandsFillGap (hmmband.c:2997). Doctor I_k's band to bridge a gap in
/// the reachable band of the target state. Only ever called with local off.
fn hmm_bands_fill_gap(cp9b: &mut CP9Bands, k: usize, min1: i32, _max1: i32, min2: i32, _max2: i32, prv_nd_r_mn: i32, prv_nd_r_dn: i32) {
    let right_min = if min1 <= min2 { min2 } else { min1 };
    let mut r#in = i32::MAX;
    if prv_nd_r_mn != i32::MAX {
        r#in = r#in.min(prv_nd_r_mn + 1);
    }
    if prv_nd_r_dn != i32::MAX {
        r#in = r#in.min(prv_nd_r_dn);
    }
    let ix = right_min - 1;
    if cp9b.pn_min_i[k] != -1 {
        cp9b.pn_min_i[k] = cp9b.pn_min_i[k].min(r#in);
    } else {
        cp9b.pn_min_i[k] = r#in;
    }
    if cp9b.pn_max_i[k] != -1 {
        cp9b.pn_max_i[k] = cp9b.pn_max_i[k].max(ix);
    } else {
        cp9b.pn_max_i[k] = ix;
    }
}

/// C: HMMBandsEnforceValidParse (hmmband.c:2270). Step 1 of cp9_HMM2ijBands.
/// Walks the HMM left->right deriving the reachable-residue r_* bands and (if
/// local is off) doctoring pn_* bands so at least one HMM parse survives.
pub fn hmm_bands_enforce_valid_parse(
    cp9: &CP9,
    cp9b: &mut CP9Bands,
    _cp9map: &CP9Map,
    i0: i32,
    j0: i32,
    doing_search: bool,
) -> RArrays {
    let hmm_m = cp9b.hmm_m;
    let sz = (hmm_m + 1) as usize;
    let local_begins_ends_on =
        (cp9.flags & CPLAN9_LOCAL_BEGIN != 0) && (cp9.flags & CPLAN9_LOCAL_END != 0);

    let mut r_mn = vec![i32::MAX; sz];
    let mut r_mx = vec![i32::MIN; sz];
    let mut r_in = vec![i32::MAX; sz];
    let mut r_ix = vec![i32::MIN; sz];
    let mut r_dn = vec![i32::MAX; sz];
    let mut r_dx = vec![i32::MIN; sz];
    let mut r_nn_i = vec![i32::MAX; sz];
    let mut r_nx_i = vec![i32::MIN; sz];
    let mut r_nn_j = vec![i32::MAX; sz];
    let mut r_nx_j = vec![i32::MIN; sz];
    let mut r_nn_hmm = vec![i32::MAX; sz];
    let mut r_nx_hmm = vec![i32::MIN; sz];
    let mut r_begn: i32 = i32::MAX;
    let mut r_begx: i32 = i32::MIN;
    let mut r_endn: i32 = i32::MAX;
    let mut r_endx: i32 = i32::MIN;

    let mut was_unr = vec![false; sz];
    let mut filled_gap = vec![false; sz];

    if cp9b.pn_min_m[0] != -1 {
        r_mn[0] = cp9b.pn_min_m[0];
        r_mx[0] = cp9b.pn_max_m[0];
    }

    let ninf = -CP9_INFTY;
    let mut k: i32 = 0;
    while k <= hmm_m {
        let ku = k as usize;
        let mut just_filled_gap = false;

        /* transitions to insert of node k (I_k) */
        if r_mn[ku] <= r_mx[ku] {
            if cp9b.pn_min_i[ku] != -1 {
                let mut n = r_mn[ku] + 1;
                let mut x = r_mx[ku] + 1;
                if x.min(cp9b.pn_max_i[ku]) - n.max(cp9b.pn_min_i[ku]) >= 0 {
                    n = n.max(cp9b.pn_min_i[ku]);
                    n = n.min(cp9b.pn_max_i[ku]);
                    x = x.min(cp9b.pn_max_i[ku]);
                    r_in[ku] = r_in[ku].min(n);
                    r_ix[ku] = r_ix[ku].max(x);
                }
            }
        }
        if r_dn[ku] <= r_dx[ku] {
            if cp9b.pn_min_i[ku] != -1 {
                let mut n = r_dn[ku] + 1;
                let mut x = r_dx[ku] + 1;
                if x.min(cp9b.pn_max_i[ku]) - n.max(cp9b.pn_min_i[ku]) >= 0 {
                    n = n.max(cp9b.pn_min_i[ku]);
                    n = n.min(cp9b.pn_max_i[ku]);
                    x = x.min(cp9b.pn_max_i[ku]);
                    r_in[ku] = r_in[ku].min(n);
                    r_ix[ku] = r_ix[ku].max(x);
                }
            }
        }
        if r_in[ku] <= r_ix[ku] {
            /* I_k -> I_k transition (self emitter) */
            if cp9b.pn_min_i[ku] != -1 {
                if r_in[ku] <= cp9b.pn_max_i[ku] {
                    r_ix[ku] = cp9b.pn_max_i[ku];
                } else {
                    r_in[ku] = i32::MAX;
                    r_ix[ku] = i32::MIN;
                }
            }
        }

        /* transitions to match of node k+1 (M_k+1) */
        if k < hmm_m {
            let kp1 = ku + 1;
            if r_mn[ku] <= r_mx[ku] {
                if cp9b.pn_min_m[kp1] != -1 {
                    let mut n = r_mn[ku] + 1;
                    let mut x = r_mx[ku] + 1;
                    if x.min(cp9b.pn_max_m[kp1]) - n.max(cp9b.pn_min_m[kp1]) >= 0 {
                        n = n.max(cp9b.pn_min_m[kp1]);
                        n = n.min(cp9b.pn_max_m[kp1]);
                        x = x.min(cp9b.pn_max_m[kp1]);
                        if r_mn[kp1] != i32::MAX
                            && !local_begins_ends_on
                            && x.min(r_mx[kp1]) - n.max(r_mn[kp1]) < -1
                        {
                            hmm_bands_fill_gap(cp9b, ku, n, x, r_mn[kp1], r_mx[kp1], r_mn[ku - 1], r_dn[ku - 1]);
                            just_filled_gap = true;
                        }
                        r_mn[kp1] = r_mn[kp1].min(n);
                        r_mx[kp1] = r_mx[kp1].max(x);
                    }
                }
            }
            /* D_k->M_k+1 transition */
            if r_dn[ku] <= r_dx[ku] {
                if cp9b.pn_min_m[kp1] != -1 {
                    let mut n = r_dn[ku] + 1;
                    let mut x = r_dx[ku] + 1;
                    if x.min(cp9b.pn_max_m[kp1]) - n.max(cp9b.pn_min_m[kp1]) >= 0 {
                        n = n.max(cp9b.pn_min_m[kp1]);
                        n = n.min(cp9b.pn_max_m[kp1]);
                        x = x.min(cp9b.pn_max_m[kp1]);
                        if r_mn[kp1] != i32::MAX
                            && !local_begins_ends_on
                            && x.min(r_mx[kp1]) - n.max(r_mn[kp1]) < -1
                        {
                            hmm_bands_fill_gap(cp9b, ku, n, x, r_mn[kp1], r_mx[kp1], r_mn[ku - 1], r_dn[ku - 1]);
                            just_filled_gap = true;
                        }
                        r_mn[kp1] = r_mn[kp1].min(n);
                        r_mx[kp1] = r_mx[kp1].max(x);
                    }
                }
            }
            /* I_k->M_k+1 transition */
            if r_in[ku] <= r_ix[ku] {
                if cp9b.pn_min_m[kp1] != -1 {
                    let mut n = r_in[ku] + 1;
                    let mut x = r_ix[ku] + 1;
                    if x.min(cp9b.pn_max_m[kp1]) - n.max(cp9b.pn_min_m[kp1]) >= 0 {
                        n = n.max(cp9b.pn_min_m[kp1]);
                        n = n.min(cp9b.pn_max_m[kp1]);
                        x = x.min(cp9b.pn_max_m[kp1]);
                        if !local_begins_ends_on && x.min(r_mx[kp1]) - n.max(r_mn[kp1]) < -1 {
                            hmm_bands_fill_gap(cp9b, ku, n, x, r_mn[kp1], r_mx[kp1], r_mn[ku - 1], r_dn[ku - 1]);
                            just_filled_gap = true;
                        }
                        r_mn[kp1] = r_mn[kp1].min(n);
                        r_mx[kp1] = r_mx[kp1].max(x);
                    }
                }
            }
            /* EL_kp->M_k+1 transition */
            if cp9.flags & CPLAN9_EL != 0 {
                if cp9b.pn_min_m[kp1] != -1 {
                    for c in 0..cp9.el_from_ct[kp1] as usize {
                        let kpp = cp9.el_from_idx[kp1][c] as usize;
                        if r_mn[kpp] <= r_mx[kpp] {
                            let mut n = r_mn[kpp];
                            let mut x = j0;
                            if x.min(cp9b.pn_max_m[kp1]) - n.max(cp9b.pn_min_m[kp1]) >= 0 {
                                n = n.max(cp9b.pn_min_m[kp1]);
                                n = n.min(cp9b.pn_max_m[kp1]);
                                x = x.min(cp9b.pn_max_m[kp1]);
                                if r_mn[kp1] != i32::MAX
                                    && !local_begins_ends_on
                                    && x.min(r_mx[kp1]) - n.max(r_mn[kp1]) < -1
                                {
                                    hmm_bands_fill_gap(cp9b, ku, n, x, r_mn[kp1], r_mx[kp1], r_mn[ku - 1], r_dn[ku - 1]);
                                    just_filled_gap = true;
                                }
                                r_mn[kp1] = r_mn[kp1].min(n);
                                r_mx[kp1] = r_mx[kp1].max(x);
                            }
                        }
                    }
                }
            }
            /* Begin ->M_k+1 transition */
            if local_begins_ends_on {
                if cp9b.pn_min_m[kp1] != -1 {
                    if doing_search {
                        let n = cp9b.pn_min_m[kp1];
                        let x = cp9b.pn_max_m[kp1];
                        r_mn[kp1] = r_mn[kp1].min(n);
                        r_mx[kp1] = r_mx[kp1].max(x);
                    } else {
                        if cp9b.pn_min_m[kp1] == r_mn[0] + 1 {
                            r_mn[kp1] = r_mn[kp1].min(r_mn[0] + 1);
                            r_mx[kp1] = r_mx[kp1].max(r_mn[0] + 1);
                        }
                    }
                }
            }
        } /* end of if(k < hmm_M) */

        /* transitions to END state */
        if k == hmm_m || local_begins_ends_on {
            if r_mn[ku] <= r_mx[ku] && cp9.esc[ku] != ninf {
                let n = r_mn[ku];
                let x = r_mx[ku];
                r_endn = r_endn.min(n);
                r_endx = r_endx.max(x);
            }
        }
        if k == hmm_m {
            if r_dn[ku] <= r_dx[ku] && cp9.tsc[ku][CTDM] != ninf {
                let n = r_dn[ku];
                let x = r_dx[ku];
                r_endn = r_endn.min(n);
                r_endx = r_endx.max(x);
            }
            if r_in[ku] <= r_ix[ku] && cp9.tsc[ku][CTIM] != ninf {
                let n = r_in[ku];
                let x = r_in[ku];
                r_endn = r_endn.min(n);
                r_endx = r_endx.max(x);
            }
            if cp9.flags & CMH_LOCAL_END != 0 {
                for c in 0..cp9.el_from_ct[ku + 1] as usize {
                    let kpp = cp9.el_from_idx[ku + 1][c] as usize;
                    if r_mn[kpp] <= r_mx[kpp] {
                        let n = r_mn[kpp];
                        let x = j0;
                        r_endn = r_endn.min(n);
                        r_endx = r_endx.max(x);
                    }
                }
            }
        }

        /* transitions to delete of node k+1 (D_k+1) */
        if k < hmm_m {
            let kp1 = ku + 1;
            /* M_k -> D_k+1 */
            if r_mn[ku] <= r_mx[ku] {
                if cp9b.pn_min_d[kp1] != -1 {
                    let mut n = r_mn[ku];
                    let mut x = r_mx[ku];
                    if x.min(cp9b.pn_max_d[kp1]) - n.max(cp9b.pn_min_d[kp1]) >= 0 {
                        n = n.max(cp9b.pn_min_d[kp1]);
                        n = n.min(cp9b.pn_max_d[kp1]);
                        x = x.min(cp9b.pn_max_d[kp1]);
                        if r_dn[kp1] != i32::MAX
                            && !local_begins_ends_on
                            && x.min(r_dx[kp1]) - n.max(r_dn[kp1]) < -1
                        {
                            hmm_bands_fill_gap(cp9b, ku, n, x, r_dn[kp1], r_dx[kp1], r_mn[ku - 1], r_dn[ku - 1]);
                            just_filled_gap = true;
                        }
                        r_dn[kp1] = r_dn[kp1].min(n);
                        r_dx[kp1] = r_dx[kp1].max(x);
                    }
                }
            }
            /* I_k -> D_k+1 */
            if r_in[ku] <= r_ix[ku] {
                if cp9b.pn_min_d[kp1] != -1 {
                    let mut n = r_in[ku];
                    let mut x = r_ix[ku];
                    if x.min(cp9b.pn_max_d[kp1]) - n.max(cp9b.pn_min_d[kp1]) >= 0 {
                        n = n.max(cp9b.pn_min_d[kp1]);
                        n = n.min(cp9b.pn_max_d[kp1]);
                        x = x.min(cp9b.pn_max_d[kp1]);
                        if r_dn[kp1] != i32::MAX
                            && !local_begins_ends_on
                            && x.min(r_dx[kp1]) - n.max(r_dn[kp1]) < -1
                        {
                            hmm_bands_fill_gap(cp9b, ku, n, x, r_dn[kp1], r_dx[kp1], r_mn[ku - 1], r_dn[ku - 1]);
                            just_filled_gap = true;
                        }
                        r_dn[kp1] = r_dn[kp1].min(n);
                        r_dx[kp1] = r_dx[kp1].max(x);
                    }
                }
            }
            /* D_k -> D_k+1 */
            if r_dn[ku] <= r_dx[ku] {
                if cp9b.pn_min_d[kp1] != -1 {
                    let mut n = r_dn[ku];
                    let mut x = r_dx[ku];
                    if x.min(cp9b.pn_max_d[kp1]) - n.max(cp9b.pn_min_d[kp1]) >= 0 {
                        n = n.max(cp9b.pn_min_d[kp1]);
                        n = n.min(cp9b.pn_max_d[kp1]);
                        x = x.min(cp9b.pn_max_d[kp1]);
                        if r_dn[kp1] != i32::MAX
                            && !local_begins_ends_on
                            && x.min(r_dx[kp1]) - n.max(r_dn[kp1]) < -1
                        {
                            hmm_bands_fill_gap(cp9b, ku, n, x, r_dn[kp1], r_dx[kp1], r_mn[ku - 1], r_dn[ku - 1]);
                            just_filled_gap = true;
                        }
                        r_dn[kp1] = r_dn[kp1].min(n);
                        r_dx[kp1] = r_dx[kp1].max(x);
                    }
                }
            }
        }

        /* update the reachable-by-node bands */
        if r_mn[ku] <= r_mx[ku] {
            r_nn_hmm[ku] = r_nn_hmm[ku].min(r_mn[ku]);
            r_nx_hmm[ku] = r_nx_hmm[ku].max(r_mx[ku]);
            let sd = 1;
            if k != hmm_m {
                r_nn_i[ku + 1] = r_nn_i[ku + 1].min(r_mn[ku] + sd);
                r_nx_i[ku + 1] = r_nx_i[ku + 1].max(r_mx[ku] + sd);
            }
            if k != 0 {
                r_nn_j[ku - 1] = r_nn_j[ku - 1].min(r_mn[ku] - sd);
                r_nx_j[ku - 1] = r_nx_j[ku - 1].max(r_mx[ku] - sd);
            }
            if (local_begins_ends_on && k > 0) || k == hmm_m {
                if doing_search {
                    r_nn_j[ku] = r_nn_j[ku].min(r_mn[ku]);
                    r_nx_j[ku] = r_nx_j[ku].max(r_mx[ku]);
                } else if r_mx[ku] == j0 {
                    r_nn_j[ku] = r_nn_j[ku].min(j0);
                    r_nx_j[ku] = r_nx_j[ku].max(j0);
                }
            }
            if (local_begins_ends_on && k > 0) || k == 1 {
                if doing_search {
                    r_nn_i[ku] = r_nn_i[ku].min(r_mn[ku]);
                    r_nx_i[ku] = r_nx_i[ku].max(r_mx[ku]);
                    r_begn = r_begn.min(r_mn[ku]);
                    r_begx = r_begx.max(r_mx[ku]);
                } else if r_mn[ku] == i0 {
                    r_nn_i[ku] = r_nn_i[ku].min(i0);
                    r_nx_i[ku] = r_nx_i[ku].max(i0);
                }
            }
        }
        if r_in[ku] <= r_ix[ku] {
            r_nn_hmm[ku] = r_nn_hmm[ku].min(r_in[ku]);
            r_nx_hmm[ku] = r_nx_hmm[ku].max(r_ix[ku]);
            let sd = 1;
            if k != hmm_m {
                r_nn_i[ku + 1] = r_nn_i[ku + 1].min(r_in[ku] + sd);
                r_nx_i[ku + 1] = r_nx_i[ku + 1].max(r_ix[ku] + sd);
            }
            r_nn_j[ku] = r_nn_j[ku].min(r_in[ku] - sd);
            r_nx_j[ku] = r_nx_j[ku].max(r_ix[ku] - sd);
            if k == 0 {
                r_begn = r_begn.min(r_in[ku]);
                r_begx = r_begx.max(r_ix[ku]);
            }
        }
        if r_dn[ku] <= r_dx[ku] {
            r_nn_hmm[ku] = r_nn_hmm[ku].min(r_dn[ku]);
            r_nx_hmm[ku] = r_nx_hmm[ku].max(r_dx[ku]);
            if k != hmm_m {
                r_nn_i[ku + 1] = r_nn_i[ku + 1].min(r_dn[ku] + 1);
                r_nx_i[ku + 1] = r_nx_i[ku + 1].max(r_dx[ku] + 1);
            }
            if k != 0 {
                r_nn_j[ku - 1] = r_nn_j[ku - 1].min(r_dn[ku]);
                r_nx_j[ku - 1] = r_nx_j[ku - 1].max(r_dx[ku]);
            }
            if k == 1 {
                r_begn = r_begn.min(r_dn[ku] + 1);
                r_begx = r_begx.max(r_dx[ku] + 1);
            }
        }

        /* is the node reachable? (only matters if local off) */
        if (!local_begins_ends_on) && (r_mn[ku] > r_mx[ku]) && (r_dn[ku] > r_dx[ku]) {
            was_unr[ku] = true;
            hmm_bands_fix_unreachable(cp9b, ku, r_nn_hmm[ku - 1], r_nx_hmm[ku - 1], r_in[ku - 1]);
            k -= 2;
        } else if just_filled_gap {
            filled_gap[ku] = true;
            k -= 1;
        }
        k += 1;
    }

    if !doing_search {
        r_begn = i0;
        r_begx = i0;
        r_endn = j0;
        r_endx = j0;
    }

    /* the r_nn_i[1] / r_nn_j[hmm_M] hack */
    r_nn_i[1] = r_nn_i[1].min(r_begn);
    r_nx_i[1] = r_nx_i[1].max(r_begx);
    r_nn_j[hmm_m as usize] = r_nn_j[hmm_m as usize].min(r_endn);
    r_nx_j[hmm_m as usize] = r_nx_j[hmm_m as usize].max(r_endx);

    for kk in 0..=hmm_m as usize {
        if r_mn[kk] == i32::MAX { r_mn[kk] = -1; }
        if r_mx[kk] == i32::MIN { r_mx[kk] = -2; }
        if r_in[kk] == i32::MAX { r_in[kk] = -1; }
        if r_ix[kk] == i32::MIN { r_ix[kk] = -2; }
        if r_dn[kk] == i32::MAX { r_dn[kk] = -1; }
        if r_dx[kk] == i32::MIN { r_dx[kk] = -2; }
        if local_begins_ends_on {
            if r_nn_i[kk] == i32::MAX { r_nn_i[kk] = -1; }
            if r_nx_i[kk] == i32::MIN { r_nx_i[kk] = -2; }
            if r_nn_j[kk] == i32::MAX { r_nn_j[kk] = -1; }
            if r_nx_j[kk] == i32::MIN { r_nx_j[kk] = -2; }
        }
    }
    let _ = (&was_unr, &filled_gap, &r_nn_hmm, &r_nx_hmm);

    RArrays { r_mn, r_mx, r_in, r_ix, r_dn, r_dx, r_nn_i, r_nx_i, r_nn_j, r_nx_j }
}

/// C: cp9_HMM2ijBands (hmmband.c:1551). Fills cp9b.imin/imax/jmin/jmax for all
/// CM states from the HMM position bands. do_trunc=FALSE path only (standard
/// non-truncated pipeline).
pub fn cp9_hmm2ijbands(
    cm: &CM,
    cp9: &CP9,
    cp9b: &mut CP9Bands,
    cp9map: &CP9Map,
    i0: i32,
    j0: i32,
    doing_search: bool,
    do_trunc: bool,
) {
    let m = cm.m as usize;
    let hmm_m = cp9b.hmm_m;
    let hmm_is_localized = (cp9.flags & CPLAN9_LOCAL_BEGIN != 0)
        || (cp9.flags & CPLAN9_LOCAL_END != 0)
        || (cp9.flags & CPLAN9_EL != 0);
    let cm_is_fully_localized =
        (cm.flags & CMH_LOCAL_BEGIN != 0) && (cm.flags & CMH_LOCAL_END != 0);

    /* Initialize all bands to -1/-2 */
    cp9b.imin = vec![-1; m];
    cp9b.imax = vec![-2; m];
    cp9b.jmin = vec![-1; m];
    cp9b.jmax = vec![-2; m];

    /* Step 1 */
    let r = hmm_bands_enforce_valid_parse(cp9, cp9b, cp9map, i0, j0, doing_search);
    let (r_mn, r_mx, r_in, r_ix, r_dn, r_dx) =
        (&r.r_mn, &r.r_mx, &r.r_in, &r.r_ix, &r.r_dn, &r.r_dx);
    let (r_nn_i, r_nx_i, r_nn_j, r_nx_j) = (&r.r_nn_i, &r.r_nx_i, &r.r_nn_j, &r.r_nx_j);

    /* Step 2: traverse the CM left to right via a stack, twice per node. */
    let mut nd: usize;
    let mut lpos: i32 = 0;
    let mut rpos: i32 = 0;
    let mut nd_pda: Vec<i32> = Vec::new();
    let mut lpos_pda: Vec<i32> = Vec::new();
    nd_pda.push(0); // 0 = left side
    nd_pda.push(0); // nd = 0

    while let Some(nd_i) = nd_pda.pop() {
        nd = nd_i as usize;
        let on_right = nd_pda.pop().unwrap();
        let ndt = cm.ndtype[nd] as i32;
        if on_right != 0 {
            match ndt {
                x if x == BIF_ND => {
                    let v = cm.nodemap[nd] as usize;
                    let w = cm.cfirst[v] as usize; // BEGL_S
                    let y = cm.cnum[v] as usize; // BEGR_S
                    cp9b.imin[v] = if cp9b.imin[w] != -1 { cp9b.imin[w] } else { cp9b.imin[y] };
                    cp9b.imax[v] = if cp9b.imax[w] != -2 { cp9b.imax[w] } else { cp9b.imax[y] };
                    cp9b.jmin[v] = if cp9b.jmin[y] != -1 { cp9b.jmin[y] } else { cp9b.jmin[w] };
                    cp9b.jmax[v] = if cp9b.jmax[y] != -2 { cp9b.jmax[y] } else { cp9b.jmax[w] };
                    if !do_trunc {
                        if cp9b.imin[w] == -1 || cp9b.jmin[w] == -1 || cp9b.imin[y] == -1 || cp9b.jmin[y] == -1
                            || cp9b.imax[w] == -2 || cp9b.imax[w] == -2 || cp9b.jmax[y] == -2 || cp9b.jmax[y] == -2
                        {
                            cp9b.imin[v] = -1; cp9b.imin[w] = -1; cp9b.imin[y] = -1;
                            cp9b.jmin[v] = -1; cp9b.jmin[w] = -1; cp9b.jmin[y] = -1;
                            cp9b.imax[v] = -2; cp9b.imax[w] = -2; cp9b.imax[y] = -2;
                            cp9b.jmax[v] = -2; cp9b.jmax[w] = -2; cp9b.jmax[y] = -2;
                            cp9b.imin[y + 1] = -1; cp9b.jmin[y + 1] = -1;
                            cp9b.imax[y + 1] = -2; cp9b.jmax[y + 1] = -2;
                        }
                    }
                }
                x if x == MATP_ND => {
                    lpos = cp9map.nd2lpos[nd];
                    rpos = cp9map.nd2rpos[nd];
                    let mut v = cm.nodemap[nd] as usize; // MATP_MP
                    cp9b.jmin[v] = r_mn[rpos as usize];
                    cp9b.jmax[v] = r_mx[rpos as usize];
                    v += 1; // MATP_ML
                    cp9b.jmin[v] = r_dn[rpos as usize];
                    cp9b.jmax[v] = r_dx[rpos as usize];
                    v += 1; // MATP_MR
                    cp9b.jmin[v] = r_mn[rpos as usize];
                    cp9b.jmax[v] = r_mx[rpos as usize];
                    v += 1; // MATP_D
                    cp9b.jmin[v] = r_dn[rpos as usize];
                    cp9b.jmax[v] = r_dx[rpos as usize];
                    v += 1; // MATP_IL
                    cp9b.jmin[v] = r_nn_j[(rpos - 1) as usize];
                    cp9b.jmax[v] = r_nx_j[(rpos - 1) as usize];
                    v += 1; // MATP_IR
                    cp9b.jmin[v] = r_in[(rpos - 1) as usize];
                    cp9b.jmax[v] = r_ix[(rpos - 1) as usize];
                    cp9b.imin[v] = r_nn_i[(lpos + 1) as usize];
                    cp9b.imax[v] = r_nx_i[(lpos + 1) as usize];
                }
                x if x == MATL_ND => {
                    lpos = cp9map.nd2lpos[nd];
                    let mut v = cm.nodemap[nd] as usize; // MATL_ML
                    cp9b.jmin[v] = r_nn_j[rpos as usize];
                    cp9b.jmax[v] = r_nx_j[rpos as usize];
                    v += 1; // MATL_D
                    cp9b.jmin[v] = r_nn_j[rpos as usize];
                    cp9b.jmax[v] = r_nx_j[rpos as usize];
                    v += 1; // MATL_IL
                    cp9b.jmin[v] = r_nn_j[rpos as usize];
                    cp9b.jmax[v] = r_nx_j[rpos as usize];
                }
                x if x == MATR_ND => {
                    rpos = cp9map.nd2rpos[nd];
                    let mut v = cm.nodemap[nd] as usize; // MATR_MR
                    cp9b.jmin[v] = r_mn[rpos as usize];
                    cp9b.jmax[v] = r_mx[rpos as usize];
                    cp9b.imin[v] = r_nn_i[lpos as usize];
                    cp9b.imax[v] = r_nx_i[lpos as usize];
                    v += 1; // MATR_D
                    cp9b.jmin[v] = r_dn[rpos as usize];
                    cp9b.jmax[v] = r_dx[rpos as usize];
                    cp9b.imin[v] = r_nn_i[lpos as usize];
                    cp9b.imax[v] = r_nx_i[lpos as usize];
                    v += 1; // MATR_IR
                    cp9b.jmin[v] = r_in[(rpos - 1) as usize];
                    cp9b.jmax[v] = r_ix[(rpos - 1) as usize];
                    cp9b.imin[v] = r_nn_i[lpos as usize];
                    cp9b.imax[v] = r_nx_i[lpos as usize];
                }
                x if x == BEGL_ND || x == BEGR_ND => {
                    let v = cm.nodemap[nd] as usize; // BEG{L,R}_S
                    cp9b.imin[v] = i32::MAX; cp9b.jmin[v] = i32::MAX;
                    cp9b.imax[v] = i32::MIN; cp9b.jmax[v] = i32::MIN;
                    let cf = cm.cfirst[v];
                    for y in cf..(cf + cm.cnum[v]) {
                        let yu = y as usize;
                        if cp9b.imin[yu] != -1 {
                            cp9b.imin[v] = cp9b.imin[v].min(cp9b.imin[yu]);
                            cp9b.imax[v] = cp9b.imax[v].max(cp9b.imax[yu]);
                        }
                        if cp9b.jmin[yu] != -1 {
                            cp9b.jmin[v] = cp9b.jmin[v].min(cp9b.jmin[yu]);
                            cp9b.jmax[v] = cp9b.jmax[v].max(cp9b.jmax[yu]);
                        }
                    }
                    if cp9b.imin[v] == i32::MAX { cp9b.imin[v] = -1; cp9b.imax[v] = -2; }
                    if cp9b.jmin[v] == i32::MAX { cp9b.jmin[v] = -1; cp9b.jmax[v] = -2; }

                    if ndt == BEGR_ND {
                        let vil = v + 1; // BEGR_IL
                        cp9b.imin[vil] = r_in[(lpos - 1) as usize];
                        cp9b.imax[vil] = r_ix[(lpos - 1) as usize];
                        if cp9b.imin[v] != -1 && cp9b.imin[vil] != -1 {
                            cp9b.imin[v] = cp9b.imin[v].min(cp9b.imin[vil]);
                            cp9b.jmin[vil] = if cp9b.jmin[v] == -1 { -1 } else { cp9b.jmin[v].max(i0) };
                            cp9b.jmax[vil] = cp9b.jmax[v];
                        } else {
                            cp9b.imin[vil] = -1; cp9b.jmin[vil] = -1;
                            cp9b.imax[vil] = -2; cp9b.jmax[vil] = -2;
                        }
                        lpos = lpos_pda.pop().unwrap();
                    } else {
                        lpos_pda.push(lpos);
                        lpos = rpos + 1;
                    }
                }
                x if x == END_ND => {
                    let v = cm.nodemap[nd] as usize; // END_E
                    cp9b.imin[v] = r_nn_i[lpos as usize];
                    cp9b.imax[v] = if r_nx_i[lpos as usize] == -2 {
                        r_nx_i[lpos as usize]
                    } else {
                        (r_nx_i[lpos as usize] + 1).min(j0 + 1)
                    };
                    if r_in[lpos as usize] != -1 {
                        cp9b.imin[v] = cp9b.imin[v].min((r_in[lpos as usize] - 1).max(i0));
                        cp9b.imax[v] = cp9b.imax[v].max((r_ix[lpos as usize] - 1).max(i0));
                    }
                    rpos = lpos;
                    if cp9b.imin[v] != -1 {
                        cp9b.jmin[v] = cp9b.imin[v] - 1;
                        cp9b.jmax[v] = cp9b.imax[v] - 1;
                    }
                }
                x if x == ROOT_ND => {
                    let mut v = cm.nodemap[nd] as usize; // ROOT_S
                    cp9b.imin[v] = r_nn_i[1];
                    cp9b.imax[v] = r_nx_i[1];
                    cp9b.jmin[v] = r_nn_j[hmm_m as usize];
                    cp9b.jmax[v] = r_nx_j[hmm_m as usize];
                    v += 1; // ROOT_IL
                    cp9b.imin[v] = r_in[0];
                    cp9b.imax[v] = r_ix[0];
                    cp9b.jmin[v] = if r_nn_j[hmm_m as usize] == -1 { -1 } else { r_nn_j[hmm_m as usize].max(i0) };
                    cp9b.jmax[v] = r_nx_j[hmm_m as usize];
                    if r_in[hmm_m as usize] != -1 {
                        cp9b.jmin[v] = cp9b.jmin[v].min(r_in[hmm_m as usize]);
                        cp9b.jmax[v] = cp9b.jmax[v].min(r_ix[hmm_m as usize]);
                    }
                    v += 1; // ROOT_IR
                    if r_in[hmm_m as usize] != -1 {
                        cp9b.imin[v] = r_nn_i[1];
                        cp9b.imax[v] = r_nx_i[1];
                        if cp9b.imin[v - 1] != -1 {
                            cp9b.imin[v] = cp9b.imin[v].min(cp9b.imin[v - 1] + 1);
                            cp9b.imax[v] = cp9b.imax[v].max(cp9b.imax[v - 1] + 1);
                        }
                        cp9b.jmin[v] = r_in[hmm_m as usize];
                        cp9b.jmax[v] = r_ix[hmm_m as usize];
                    }
                }
                _ => {}
            } /* end switch on_right */
        } else {
            /* on left: set i bands for MATP_nd, MATL_nd only */
            match ndt {
                x if x == MATP_ND => {
                    lpos = cp9map.nd2lpos[nd];
                    let mut v = cm.nodemap[nd] as usize; // MATP_MP
                    cp9b.imin[v] = r_mn[lpos as usize];
                    cp9b.imax[v] = r_mx[lpos as usize];
                    v += 1; // MATP_ML
                    cp9b.imin[v] = r_mn[lpos as usize];
                    cp9b.imax[v] = r_mx[lpos as usize];
                    v += 1; // MATP_MR
                    cp9b.imin[v] = if r_dn[lpos as usize] == -1 { -1 } else { r_dn[lpos as usize] + 1 };
                    cp9b.imax[v] = if r_dx[lpos as usize] == -2 { -2 } else { r_dx[lpos as usize] + 1 };
                    v += 1; // MATP_D
                    cp9b.imin[v] = if r_dn[lpos as usize] == -1 { -1 } else { r_dn[lpos as usize] + 1 };
                    cp9b.imax[v] = if r_dx[lpos as usize] == -2 { -2 } else { r_dx[lpos as usize] + 1 };
                    v += 1; // MATP_IL
                    cp9b.imin[v] = r_in[lpos as usize];
                    cp9b.imax[v] = r_ix[lpos as usize];
                }
                x if x == MATL_ND => {
                    lpos = cp9map.nd2lpos[nd];
                    let mut v = cm.nodemap[nd] as usize; // MATL_ML
                    cp9b.imin[v] = r_mn[lpos as usize];
                    cp9b.imax[v] = r_mx[lpos as usize];
                    v += 1; // MATL_D
                    cp9b.imin[v] = if r_dn[lpos as usize] == -1 { -1 } else { r_dn[lpos as usize] + 1 };
                    cp9b.imax[v] = if r_dx[lpos as usize] == -2 { -2 } else { r_dx[lpos as usize] + 1 };
                    v += 1; // MATL_IL
                    cp9b.imin[v] = r_in[lpos as usize];
                    cp9b.imax[v] = r_ix[lpos as usize];
                }
                _ => {}
            }

            if ndt == BIF_ND {
                nd_pda.push(1);
                nd_pda.push(nd as i32);
                /* right child */
                nd_pda.push(0);
                nd_pda.push(cm.ndidx[cm.cnum[cm.nodemap[nd] as usize] as usize]);
                /* left child */
                nd_pda.push(0);
                nd_pda.push(cm.ndidx[cm.cfirst[cm.nodemap[nd] as usize] as usize]);
            } else {
                nd_pda.push(1);
                nd_pda.push(nd as i32);
                if ndt != END_ND {
                    nd_pda.push(0);
                    nd_pda.push(nd as i32 + 1);
                }
            }
        }
    }

    /* do_trunc final pass skipped (do_trunc == false in standard pipeline) */

    if !doing_search {
        cp9b.imin[0] = i0;
        if cp9b.imin[1] != -1 { cp9b.imin[1] = i0; }
        cp9b.jmax[0] = j0;
        if cp9b.jmin[1] != -1 { cp9b.jmax[1] = j0; }
        if cp9b.jmin[2] != -1 { cp9b.jmax[2] = j0; }
    }

    /* Final pass through all states */
    for v in 0..m {
        if cp9b.imin[v] == -1 || cp9b.jmin[v] == -1 {
            cp9b.imin[v] = -1; cp9b.jmin[v] = -1;
            cp9b.imax[v] = -2; cp9b.jmax[v] = -2;
        }
        if state_is_detached(cm, v) {
            cp9b.imin[v] = -1; cp9b.jmin[v] = -1;
            cp9b.imax[v] = -2; cp9b.jmax[v] = -2;
        }
        if !do_trunc {
            if cm.sttype[v] as i32 == crate::constants::MP_ST {
                if cp9b.jmax[v] == i0 {
                    cp9b.imin[v] = -1; cp9b.jmin[v] = -1;
                    cp9b.imax[v] = -2; cp9b.jmax[v] = -2;
                } else if cp9b.jmin[v] == i0 {
                    cp9b.jmin[v] += 1;
                }
            }
            if state_left_delta(cm.sttype[v] as i32) == 1 && cp9b.imin[v] != -1 {
                if cp9b.jmax[v] == i0 - 1 {
                    cp9b.imin[v] = -1; cp9b.jmin[v] = -1;
                    cp9b.imax[v] = -2; cp9b.jmax[v] = -2;
                } else if cp9b.jmin[v] == i0 - 1 {
                    cp9b.jmin[v] = i0;
                }
            }
        }
    }

    /* brutal hack: localized case, ensure at least one valid parse (search branch) */
    if hmm_is_localized && cm_is_fully_localized {
        if cp9b.imin[0] == -1 {
            cp9b.imin[0] = i0; cp9b.imax[0] = i0;
            cp9b.jmin[0] = j0; cp9b.jmax[0] = j0;
        }
        let mut nd1: usize = 1;
        if i0 == j0 {
            while nd1 < cm.nodes as usize && cm.ndtype[nd1] as i32 == MATP_ND {
                nd1 += 1;
            }
        }
        if cm.ndtype[nd1] as i32 == BIF_ND {
            let v = cm.nodemap[nd1] as usize;
            let w = cm.cfirst[v] as usize; // BEGL_S
            let y = cm.cnum[v] as usize; // BEGR_S
            if cp9b.imin[v] != -1 && cp9b.imin[w] != -1 && cp9b.imin[y] != -1 {
                if !doing_search {
                    cp9b.imin[v] = cp9b.imin[v].min(i0);
                    cp9b.imax[v] = cp9b.imax[v].max(i0);
                    cp9b.jmin[v] = cp9b.jmin[v].min(j0);
                    cp9b.jmax[v] = cp9b.jmax[v].max(j0);
                    cp9b.imin[w] = cp9b.imin[v];
                    cp9b.imax[w] = cp9b.imax[v];
                    cp9b.jmax[w] = cp9b.jmax[w].max(j0.min(cp9b.imax[w]));
                    cp9b.jmin[y] = cp9b.jmin[v];
                    cp9b.jmax[y] = cp9b.jmax[v];
                    cp9b.imin[y] = cp9b.imin[y].min(i0.max(cp9b.jmin[y]));
                    cp9b.imin[y] = cp9b.imin[y].min(i0.max(cp9b.jmax[w] + 1));
                    cp9b.imax[y] = cp9b.imin[y].max(cp9b.imax[y]);
                } else {
                    cp9b.imin[y] = cp9b.imin[y].min(i0.max(cp9b.jmax[w] + 1));
                    cp9b.imax[y] = cp9b.imin[y].max(cp9b.imax[y]);
                }
            } else {
                if !doing_search {
                    cp9b.imin[v] = i0; cp9b.imax[v] = i0;
                    cp9b.jmin[v] = j0; cp9b.jmax[v] = j0;
                    cp9b.imin[w] = i0; cp9b.imax[w] = i0;
                    cp9b.jmin[w] = j0 - 1; cp9b.jmax[w] = j0 - 1;
                    cp9b.imin[y] = j0; cp9b.imax[y] = j0;
                    cp9b.jmin[y] = j0; cp9b.jmax[y] = j0;
                } else {
                    cp9b.imin[v] = cp9b.imin[0]; cp9b.imax[v] = cp9b.imin[0];
                    cp9b.jmin[v] = cp9b.jmax[0]; cp9b.jmax[v] = cp9b.jmax[0];
                    cp9b.imin[w] = cp9b.imin[0]; cp9b.imax[w] = cp9b.imin[0];
                    cp9b.jmin[w] = cp9b.jmax[0] - 1; cp9b.jmax[w] = cp9b.jmax[0] - 1;
                    cp9b.imin[y] = cp9b.jmax[0]; cp9b.imax[y] = cp9b.jmax[0];
                    cp9b.jmin[y] = cp9b.jmax[0]; cp9b.jmax[y] = cp9b.jmax[0];
                }
            }
        } else {
            /* node nd1 is a MATL, MATR or MATP */
            let v = cm.nodemap[nd1] as usize;
            if !doing_search {
                if cp9b.imin[v] == -1 {
                    cp9b.imin[v] = i0; cp9b.imax[v] = i0;
                    cp9b.jmin[v] = j0; cp9b.jmax[v] = j0;
                } else {
                    cp9b.imin[v] = cp9b.imin[v].min(i0);
                    cp9b.imax[v] = cp9b.imax[v].max(i0);
                    cp9b.jmin[v] = cp9b.jmin[v].min(j0);
                    cp9b.jmax[v] = cp9b.jmax[v].max(j0);
                }
            } else {
                if cp9b.imin[v] == -1 {
                    cp9b.imin[v] = cp9b.imin[0];
                    cp9b.imax[v] = cp9b.imax[0];
                    cp9b.jmin[v] = cp9b.jmin[0];
                    cp9b.jmax[v] = cp9b.jmax[0];
                } else {
                    cp9b.imin[0] = cp9b.imin[0].min(cp9b.imin[v]);
                    cp9b.imax[0] = cp9b.imax[0].max(cp9b.imax[v]);
                    cp9b.jmin[0] = cp9b.jmin[0].min(cp9b.jmin[v]);
                    cp9b.jmax[0] = cp9b.jmax[0].max(cp9b.jmax[v]);
                }
            }
        }
    }
    let _ = hmm_is_localized;
}

/// C: StateRightDelta (cm.c). # residues emitted to the right by a state.
fn state_right_delta(sttype: i32) -> i32 {
    match sttype {
        crate::constants::D_ST => 0,
        crate::constants::MP_ST => 1,
        crate::constants::ML_ST => 0,
        crate::constants::MR_ST => 1,
        IL_ST => 0,
        IR_ST => 1,
        S_ST => 0,
        E_ST => 0,
        crate::constants::B_ST => 0,
        EL_ST => 0,
        _ => panic!("bogus state type {}", sttype),
    }
}

/// C: ij2d_bands (hmmband.c:1443). Derive the d band (subsequence length) for
/// each state v at each j in its band, from the i/j bands. Fills cp9b.hdmin/hdmax.
/// do_trunc=FALSE path only.
pub fn ij2d_bands(cm: &CM, _w: i32, cp9b: &mut CP9Bands, do_trunc: bool) {
    let m = cm.m as usize;
    cp9b.hdmin = vec![Vec::new(); m];
    cp9b.hdmax = vec![Vec::new(); m];
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        let jbw = cp9b.jmax[v] - cp9b.jmin[v]; // may be -1 (unreachable)
        let n = if jbw >= 0 { (jbw + 1) as usize } else { 0 };
        let mut hdmin = vec![0i32; n];
        let mut hdmax = vec![0i32; n];
        if stt == E_ST {
            for jp in 0..n {
                hdmin[jp] = 0;
                hdmax[jp] = 0;
            }
        } else {
            let sd = state_delta(stt);
            let max_sdl_sdr = state_left_delta(stt).max(state_right_delta(stt));
            let dn = if do_trunc { max_sdl_sdr } else { sd };
            for jp in 0..n {
                let j = jp as i32 + cp9b.jmin[v];
                let hdn = j - cp9b.imax[v] + 1;
                let hdx = j - cp9b.imin[v] + 1;
                if hdx < dn {
                    hdmin[jp] = -1;
                    hdmax[jp] = -2;
                } else {
                    hdmin[jp] = hdn.max(dn);
                    hdmax[jp] = hdx;
                }
            }
        }
        cp9b.hdmin[v] = hdmin;
        cp9b.hdmax[v] = hdmax;
    }
}

/// C: cm_hb_mx_SizeNeeded (cm_mx.c:1146). Approximate size (Mb) of the HMM
/// banded CM DP matrix given the current bands. Returns (ncells, Mb).
pub fn cm_hb_mx_size_needed(cm: &CM, cp9b: &CP9Bands, l: i32) -> (i64, f32) {
    let cm_m = cm.m; // cp9b->cm_M == cm->M
    let have_el = cm.flags & CMH_LOCAL_END != 0;
    let mut ncells: i64 = 0;
    // sizeof(CM_HB_MX)=64, sizeof(float**)=8, sizeof(int)=4, sizeof(float*)=8 (64-bit)
    let mut mb_needed: f32 = (64
        + ((cm_m + 1) as usize * 8)
        + ((cm_m + 1) as usize * 4)) as f32;
    for v in 0..cm_m as usize {
        let jbw = cp9b.jmax[v] - cp9b.jmin[v];
        mb_needed += (8 * (jbw + 1)) as f32;
        for jp in 0..=jbw {
            ncells += (cp9b.hdmax[v][jp as usize] - cp9b.hdmin[v][jp as usize] + 1) as i64;
        }
    }
    if have_el {
        ncells += ((l + 2) as f64 * (l + 1) as f64 * 0.5) as i64;
    }
    mb_needed += 4.0 * ncells as f32;
    mb_needed *= 0.000001;
    (ncells, mb_needed)
}

/// C: cp9_Seq2Bands (hmmband.c:226), default (non-truncated, do_old_hmm2ij=FALSE,
/// use_sums=FALSE) path. Runs cp9 Forward/Backward -> HMM bands -> CM ij bands
/// -> d bands. Fills cp9b. Returns the CP9 fwd/bck matrices for reuse if needed.
pub fn cp9_seq2bands(
    cm: &CM,
    cp9: &CP9,
    cp9map: &CP9Map,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    tau: f64,
    doing_search: bool,
) -> CP9Bands {
    let do_fwd_scan = doing_search; // non-truncated search: scan both
    let (_fs, _fp, fmx, _fa) = cp9_forward(cp9, dsq, i0 as usize, j0 as usize, do_fwd_scan, !doing_search);
    let (_bs, _bp, bmx, _ba) = cp9_backward(cp9, dsq, i0 as usize, j0 as usize, do_fwd_scan, !doing_search);
    let mut cp9b = cp9_fb2hmmbands(cp9, dsq, &fmx, &bmx, i0 as usize, j0 as usize, 1.0 - tau, do_fwd_scan);
    cp9_hmm2ijbands(cm, cp9, &mut cp9b, cp9map, i0, j0, doing_search, false);
    ij2d_bands(cm, j0 - i0 + 1, &mut cp9b, false);
    cp9b
}

/// C: cp9_IterateSeq2Bands (hmmband.c:413), non-truncated search path. Doubles
/// tau by TAU_MULTIPLIER until the banded matrix fits `size_limit` Mb (or tau
/// hits maxtau). Returns (bands, final_tau, hbmx_Mb).
pub fn cp9_iterate_seq2bands(
    cm: &CM,
    cp9: &CP9,
    cp9map: &CP9Map,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    start_tau: f64,
    maxtau: f64,
    size_limit: f32,
    doing_search: bool,
    do_iterate: bool,
) -> (CP9Bands, f64, f32) {
    const TAU_MULTIPLIER: f64 = 2.0;
    let mut tau = start_tau;
    let mut tau_at_limit = false;
    loop {
        let cp9b = cp9_seq2bands(cm, cp9, cp9map, dsq, i0, j0, tau, doing_search);
        let (_nc, hbmx_mb) = cm_hb_mx_size_needed(cm, &cp9b, j0 - i0 + 1);
        if hbmx_mb < size_limit || !do_iterate || tau_at_limit {
            return (cp9b, tau, hbmx_mb);
        }
        if !tau_at_limit {
            tau *= TAU_MULTIPLIER;
            if tau >= maxtau {
                tau = maxtau;
                tau_at_limit = true;
            }
        }
    }
}

// ===========================================================================
// (d2a) CM local config + log-odds. Produces the bits scores F6 CYK consumes:
//       cm.tsc / cm.esc / cm.oesc / cm.beginsc / cm.endsc.  Fresh transcription
//       of cm_localize (cm_modelconfig.c:467) + CMLogoddsify (cm.c:784) +
//       FCalcOptimizedEmitScores (cm.c:3604), canonical residues only (degenerate
//       oesc entries deferred, like CP9 msc/isc — only matter if a window has N).
// ===========================================================================

const CM_IMPOSSIBLE: f32 = -1e36;

/// C: cm_localize (cm_modelconfig.c:467). Spread p_internal_start over internal
/// node begins, zero t[0] (saving root_trans), spread p_internal_exit over
/// internal-node ends and renormalize each affected t[v]. Sets local flags.
pub fn cm_config_local(cm: &mut CM) {
    let p_internal_start = cm.pbegin;
    let p_internal_exit = cm.pend;

    // cm_CalculateLocalBeginProbs
    let mut nstarts = 0;
    for nd in 2..cm.nodes as usize {
        let t = cm.ndtype[nd] as i32;
        if t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BIF_ND {
            nstarts += 1;
        }
    }
    for v in 0..cm.m as usize {
        cm.begin[v] = 0.0;
    }
    cm.begin[cm.nodemap[1] as usize] = 1.0 - p_internal_start;
    let p = p_internal_start / nstarts as f32;
    for nd in 2..cm.nodes as usize {
        let t = cm.ndtype[nd] as i32;
        if t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BIF_ND {
            cm.begin[cm.nodemap[nd] as usize] = p;
        }
    }

    // Erase node-0 transitions (save first)
    let cnum0 = cm.cnum[0] as usize;
    if cm.root_trans.is_none() {
        cm.root_trans = Some(cm.t[0][..cnum0].to_vec());
    }
    for y in 0..cnum0 {
        cm.t[0][y] = 0.0;
    }
    cm.flags |= 1 << 10; // CMH_LOCAL_BEGIN

    // Local ends. C relies on && short-circuit: ndtype[nd+1] is only read when
    // the first clause is true, so it never indexes past the last node.
    let is_exit_node = |cm: &CM, nd: usize| -> bool {
        let t = cm.ndtype[nd] as i32;
        (t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BEGL_ND || t == BEGR_ND)
            && cm.ndtype[nd + 1] as i32 != END_ND
    };
    let mut nexits = 0;
    for nd in 1..cm.nodes as usize {
        if is_exit_node(cm, nd) {
            nexits += 1;
        }
    }
    for v in 0..cm.m as usize {
        cm.end[v] = 0.0;
    }
    for nd in 1..cm.nodes as usize {
        if is_exit_node(cm, nd) {
            let v = cm.nodemap[nd] as usize;
            cm.end[v] = p_internal_exit / nexits as f32;
            let cnum = cm.cnum[v] as usize;
            let mut denom = esl_vec_fsum(&cm.t[v][..cnum]);
            denom += cm.end[v];
            let inv = 1.0 / denom;
            for x in 0..cnum {
                cm.t[v][x] *= inv;
            }
        }
    }
    cm.flags |= 1 << 11; // CMH_LOCAL_END
    cm.flags &= !(1 << 0); // invalidate CMH_BITS (bit0 here is a placeholder; not used)
}

/// C: CMLogoddsify (cm.c:784) + FCalcOptimizedEmitScores (cm.c:3604), canonical
/// residues only. Fills cm.tsc/esc/oesc/beginsc/endsc from the (localized) probs.
pub fn cm_logoddsify(cm: &mut CM) {
    let k = ALPHABET_SIZE; // 4
    let kp = crate::cm::ALPHABET_SIZE_P; // 18
    let m = cm.m as usize;
    // ensure per-state score vectors exist (reader may leave them empty)
    if cm.tsc.len() < m { cm.tsc.resize(m, Vec::new()); }
    if cm.esc.len() < m { cm.esc.resize(m, Vec::new()); }
    if cm.oesc.len() < m { cm.oesc.resize(m, Vec::new()); }
    if cm.beginsc.len() < m { cm.beginsc.resize(m, 0.0); }
    if cm.endsc.len() < m { cm.endsc.resize(m, 0.0); }
    for v in 0..cm.m as usize {
        let stt = cm.sttype[v] as i32;
        // transitions (not B/E)
        if stt != crate::constants::B_ST && stt != E_ST {
            let cnum = cm.cnum[v] as usize;
            let mut tsc = vec![0.0f32; cnum];
            for x in 0..cnum {
                tsc[x] = sre_log2(cm.t[v][x] as f64) as f32;
            }
            cm.tsc[v] = tsc;
        }
        // emissions
        if stt == crate::constants::MP_ST {
            let mut esc = vec![0.0f32; k * k];
            for x in 0..k {
                for y in 0..k {
                    let arg = (cm.e[v][x * k + y] / (cm.null[x] * cm.null[y])) as f64;
                    esc[x * k + y] = sre_log2(arg) as f32;
                }
            }
            // oesc: Kp*Kp. C FCalcOptimizedEmitScores: init IMPOSSIBLE, canonical
            // a,b<K = esc[a*K+b]; degenerate marginals over the fraction vectors
            // (Easel FCount); any gap/missing stays IMPOSSIBLE.
            let mut oesc = vec![CM_IMPOSSIBLE; kp * kp];
            for a in 0..k {
                for b in 0..k {
                    oesc[a * kp + b] = esc[a * k + b];
                }
            }
            // a degenerate (K+1..Kp-1), b canonical: FastPairScoreLeftOnlyDegenerate
            for a in (k + 1)..(kp - 1) {
                for b in 0..k {
                    let mut sc = 0.0f32;
                    for l in 0..k {
                        sc += esc[l * k + b] * crate::cm::abc_fcount_frac(a, l);
                    }
                    oesc[a * kp + b] = sc;
                }
            }
            // a canonical, b degenerate: FastPairScoreRightOnlyDegenerate
            for a in 0..k {
                for b in (k + 1)..(kp - 1) {
                    let mut sc = 0.0f32;
                    for r in 0..k {
                        sc += esc[a * k + r] * crate::cm::abc_fcount_frac(b, r);
                    }
                    oesc[a * kp + b] = sc;
                }
            }
            // both degenerate: FastPairScoreBothDegenerate
            for a in (k + 1)..(kp - 1) {
                for b in (k + 1)..(kp - 1) {
                    let mut sc = 0.0f32;
                    for l in 0..k {
                        for r in 0..k {
                            sc += esc[l * k + r]
                                * crate::cm::abc_fcount_frac(a, l)
                                * crate::cm::abc_fcount_frac(b, r);
                        }
                    }
                    oesc[a * kp + b] = sc;
                }
            }
            cm.esc[v] = esc;
            cm.oesc[v] = oesc;
        } else if stt == crate::constants::ML_ST
            || stt == crate::constants::MR_ST
            || stt == IL_ST
            || stt == IR_ST
        {
            let mut esc = vec![0.0f32; k];
            for x in 0..k {
                let arg = (cm.e[v][x] / cm.null[x]) as f64;
                esc[x] = sre_log2(arg) as f32;
            }
            // oesc: Kp. C FCalcOptimizedEmitScores: canonical a<K = esc[a];
            // gap[K]=IMPOSSIBLE; degenerate (K+1..Kp-1) = uniform FAvgScore;
            // missing[Kp-1]=IMPOSSIBLE.
            let mut oesc = vec![CM_IMPOSSIBLE; kp];
            for a in 0..k {
                oesc[a] = esc[a];
            }
            for a in (k + 1)..(kp - 1) {
                oesc[a] = crate::cm::abc_favg_score(a, &esc);
            }
            cm.esc[v] = esc;
            cm.oesc[v] = oesc;
        }
        cm.beginsc[v] = sre_log2(cm.begin[v] as f64) as f32;
        cm.endsc[v] = sre_log2(cm.end[v] as f64) as f32;
    }
    cm.flags |= 1 << 12; // CMH_BITS (placeholder marker)
}

/// C: the score-relevant portion of cm_ConfigureSub (cm_modelconfig.c) for the
/// standard (non-truncated) search: localize the CM then logoddsify. The CP9 is
/// assumed already built on the GLOBAL cm before this is called.
pub fn cm_configure_scores(cm: &mut CM) {
    cm_config_local(cm);
    // el_selfsc adjustment (cm_modelconfig.c:342): only triggers for huge W.
    if (cm.el_selfsc * cm.w as f32) < CM_IMPOSSIBLE {
        cm.el_selfsc = CM_IMPOSSIBLE / (cm.w as f32 + 1.0);
    }
    cm_logoddsify(cm);
}

// ===========================================================================
// (d2b) F6: FastCYKScanHB (cm_dpsearch.c:3274) — HMM-banded CYK scan.
//       FILTER path only (hitlist == NULL): returns (vsc_root, envi, envj).
//       No gamma/tmp_hitlist/null3 (those only affect hit reporting, skipped
//       when hitlist is NULL). Consumes the verified cm scores + cp9 bands.
// ===========================================================================

// Reusable per-thread banded DP matrix, shared by F6 CYK and F7 Inside (they run
// sequentially on a thread, one envelope at a time). Taken out at entry, resized
// + refilled to CM_IMPOSSIBLE (correctness unchanged), returned before exit —
// avoids a fresh heap alloc of the whole band volume on every envelope.
thread_local! {
    static HB_ALPHA: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// C: FastCYKScanHB (cm_dpsearch.c:3274), hitlist==NULL filter path. Returns
/// (vsc_root, envi, envj); envi=-1/envj=-1 if no cell >= env_cutoff.
// CYK inner d-loop: alpha[iv0+k] = max(alpha[iv0+k], alpha[iy0+k] + tsc) for k<n.
// Contiguous over d (band already clamped to the child∩parent overlap, so no
// per-lane masking). This is max-plus: `_mm256_max_ps(vv, cand)` returns
// (vv < cand) ? cand : vv — byte-for-byte identical to the scalar
// `if cand > *av { *av = cand }` (incl. tie and ±0 cases), so the result is
// bit-identical to scalar, not merely close. Parent (v) and child (y) rows are
// disjoint, so the load/store windows never alias.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cyk_max_add_avx2(alpha: &mut [f32], iv0: usize, iy0: usize, n: usize, tsc: f32) {
    use std::arch::x86_64::*;
    let tv = _mm256_set1_ps(tsc);
    let mut k = 0usize;
    while k + 8 <= n {
        let cand = _mm256_add_ps(_mm256_loadu_ps(alpha.as_ptr().add(iy0 + k)), tv);
        let vv = _mm256_loadu_ps(alpha.as_ptr().add(iv0 + k));
        _mm256_storeu_ps(alpha.as_mut_ptr().add(iv0 + k), _mm256_max_ps(vv, cand));
        k += 8;
    }
    while k < n {
        let cand = *alpha.get_unchecked(iy0 + k) + tsc;
        let av = alpha.get_unchecked_mut(iv0 + k);
        if cand > *av {
            *av = cand;
        }
        k += 1;
    }
}

pub fn fast_cyk_scan_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    env_cutoff: f32,
) -> (f32, i64, i64) {
    let m = cm.m as usize;
    let kp = crate::cm::ALPHABET_SIZE_P; // 18
    #[cfg(target_arch = "x86_64")]
    let cyk_use_avx2 = is_x86_feature_detected!("avx2");
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let w = (j0 - i0 + 1) as usize;

    // --- Build flat banded matrix layout: roff(v,jp) = flat index of (v,jp,dp=0) ---
    // roff_flat is one contiguous table (vs Vec<Vec>): roff_start[v] is v's base,
    // roff_flat[roff_start[v] + jp] the row offset. Fewer heap blocks, better locality.
    let nj = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut roff_start: Vec<usize> = vec![0; m];
    let mut roff_flat: Vec<usize> = Vec::new();
    let mut ncells: usize = 0;
    for v in 0..m {
        let njv = nj(v);
        roff_start[v] = roff_flat.len();
        for jp in 0..njv {
            roff_flat.push(ncells);
            let width = hdmax[v][jp] - hdmin[v][jp] + 1; // >=0 by -1/-2 convention
            if width > 0 {
                ncells += width as usize;
            }
        }
    }
    let mut alpha = HB_ALPHA.with(|c| std::mem::take(&mut *c.borrow_mut()));
    alpha.clear();
    alpha.resize(ncells, CM_IMPOSSIBLE);

    // el_scA[d] = el_selfsc * d
    let mut el_sca = vec![0.0f32; w + 1];
    for d in 0..=w {
        el_sca[d] = cm.el_selfsc * d as f32;
    }

    let ninf = CM_IMPOSSIBLE;

    // --- Main recursion: v = M-1 .. 0 ---
    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        // Hoist the v-indexed rows out of the inner loops: one bounds-checked
        // slice deref here instead of a Vec<Vec> double-index every iteration.
        // (byte-identical: same elements, checks still on.)
        let roff_v = &roff_flat[roff_start[v]..];
        let hdmin_v = &hdmin[v];
        let hdmax_v = &hdmax[v];
        let oesc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];

        // re-init deck if we can do a local end from v
        if cm.endsc[v] > -9.999e35 {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut dp_v = 0usize;
                let mut d = hdmin_v[jp_v];
                while d <= hdmax_v[jp_v] {
                    let dp = (d - sd).max(0);
                    unsafe { *alpha.get_unchecked_mut(roff_v[jp_v] + dp_v) = el_sca[dp as usize] + cm.endsc[v]; }
                    dp_v += 1;
                    d += 1;
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                unsafe { *alpha.get_unchecked_mut(roff_v[jp_v]) = 0.0; }
            }
        } else if stt == IL_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                // valid children (self-transit allowed)
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let base_v = roff_v[jp_v];
                let hdmin_vjp = hdmin_v[jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax_v[jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut best = unsafe { *alpha.get_unchecked(base_v + dp_v) };
                    for &yoffset in &yvalid {
                        let y = cfirst + yoffset;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            let cand = unsafe { *alpha.get_unchecked(roff_flat[roff_start[y] + jp_y_sdr] + dp_y_sd) } + tsc_v[yoffset];
                            if cand > best {
                                best = cand;
                            }
                        }
                    }
                    best += oesc_v[dsq[i as usize] as usize];
                    if best < ninf {
                        best = ninf;
                    }
                    unsafe { *alpha.get_unchecked_mut(base_v + dp_v) = best; }
                    d += 1;
                }
            }
        } else if stt == IR_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let base_v = roff_v[jp_v];
                let hdmin_vjp = hdmin_v[jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax_v[jp_v] {
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut best = unsafe { *alpha.get_unchecked(base_v + dp_v) };
                    for &yoffset in &yvalid {
                        let y = cfirst + yoffset;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            let cand = unsafe { *alpha.get_unchecked(roff_flat[roff_start[y] + jp_y_sdr] + dp_y_sd) } + tsc_v[yoffset];
                            if cand > best {
                                best = cand;
                            }
                        }
                    }
                    best += oesc_v[dsq[j as usize] as usize];
                    if best < ninf {
                        best = ninf;
                    }
                    unsafe { *alpha.get_unchecked_mut(base_v + dp_v) = best; }
                    d += 1;
                }
            }
        } else if stt != crate::constants::B_ST {
            // ML, MP, MR, D, S: no self-transit; for y { for j { for d } }
            for yoffset in 0..cnum {
                let y = cfirst + yoffset;
                let tsc = tsc_v[yoffset];
                let roff_y = &roff_flat[roff_start[y]..];
                let hdmin_y = &hdmin[y];
                let hdmax_y = &hdmax[y];
                let jn = jmin[v].max(jmin[y] + sdr);
                let jx = jmax[v].min(jmax[y] + sdr);
                let mut jp_v = (jn - jmin[v]) as i32;
                let mut jp_y_sdr = (jn - jmin[y] - sdr) as i32;
                let mut jj = jn;
                while jj <= jx {
                    let jpv = jp_v as usize;
                    let jpysdr = jp_y_sdr as usize;
                    let dn = hdmin_v[jpv].max(hdmin_y[jpysdr] + sd);
                    let dx = hdmax_v[jpv].min(hdmax_y[jpysdr] + sd);
                    let mut dp_v = (dn - hdmin_v[jpv]) as i32;
                    let mut dp_y_sd = (dn - hdmin_y[jpysdr] - sd) as i32;
                    let base_v = roff_v[jpv];
                    let base_y = roff_y[jpysdr];
                    // iv,iy proven < ncells by band construction (checked path
                    // verified byte-identical); elide the bounds checks.
                    if dx >= dn {
                        let n = (dx - dn + 1) as usize;
                        let iv0 = base_v + dp_v as usize;
                        let iy0 = base_y + dp_y_sd as usize;
                        #[cfg(target_arch = "x86_64")]
                        {
                            if cyk_use_avx2 {
                                unsafe { cyk_max_add_avx2(&mut alpha[..], iv0, iy0, n, tsc) };
                            } else {
                                for k in 0..n {
                                    let cand = unsafe { *alpha.get_unchecked(iy0 + k) } + tsc;
                                    unsafe {
                                        let av = alpha.get_unchecked_mut(iv0 + k);
                                        if cand > *av { *av = cand; }
                                    }
                                }
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        for k in 0..n {
                            let cand = unsafe { *alpha.get_unchecked(iy0 + k) } + tsc;
                            unsafe {
                                let av = alpha.get_unchecked_mut(iv0 + k);
                                if cand > *av { *av = cand; }
                            }
                        }
                    }
                    let _ = (dp_v, dp_y_sd);
                    jp_v += 1;
                    jp_y_sdr += 1;
                    jj += 1;
                }
            }
            // add emission scores
            if stt == crate::constants::ML_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin_v[jp_v] + 1;
                    let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                    let base_v = roff_v[jp_v];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        unsafe { *alpha.get_unchecked_mut(base_v + dp_v as usize) += oesc_v[dsq[i as usize] as usize]; }
                        i -= 1;
                    }
                }
            } else if stt == crate::constants::MR_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                    let base_v = roff_v[jp_v];
                    let em = oesc_v[dsq[j as usize] as usize];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        unsafe { *alpha.get_unchecked_mut(base_v + dp_v as usize) += em; }
                    }
                }
            } else if stt == crate::constants::MP_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin_v[jp_v] + 1;
                    let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                    let base_v = roff_v[jp_v];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                        unsafe { *alpha.get_unchecked_mut(base_v + dp_v as usize) += oesc_v[idx]; }
                        i -= 1;
                    }
                }
            }
            // clamp to IMPOSSIBLE
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                let base_v = roff_v[jp_v];
                for dp_v in 0..=width.max(-1) {
                    if width < 0 { break; }
                    let idx = base_v + dp_v as usize;
                    if alpha[idx] < ninf {
                        alpha[idx] = ninf;
                    }
                }
            }
        } else {
            // B_st
            let y = cfirst; // left subtree
            let z = cnum; // right subtree
            let roff_y = &roff_flat[roff_start[y]..];
            let hdmin_y = &hdmin[y];
            let hdmax_y = &hdmax[y];
            let roff_z = &roff_flat[roff_start[z]..];
            let hdmin_z = &hdmin[z];
            let hdmax_z = &hdmax[z];
            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let hdmin_zjp = hdmin_z[jp_z];
                let kn0 = (j - jmax[y]).max(hdmin_zjp);
                let kn = kn0.max(0);
                let kx = jp_y.min(hdmax_z[jp_z]);
                let base_v = roff_v[jp_v];
                let base_z = roff_z[jp_z];
                let hdmin_vjp = hdmin_v[jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax_v[jp_v] {
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut best = unsafe { *alpha.get_unchecked(base_v + dp_v) };
                    let mut k = kn;
                    while k <= kx {
                        let jp_y_mk = (jp_y - k) as usize;
                        if k >= d - hdmax_y[jp_y_mk] && k <= d - hdmin_y[jp_y_mk] {
                            let kp_z = (k - hdmin_zjp) as usize;
                            let dp_y = d - hdmin_y[jp_y_mk];
                            let iy = roff_y[jp_y_mk] + (dp_y - k) as usize;
                            let iz = base_z + kp_z;
                            let cand = unsafe { *alpha.get_unchecked(iy) + *alpha.get_unchecked(iz) };
                            if cand > best {
                                best = cand;
                            }
                        }
                        k += 1;
                    }
                    unsafe { *alpha.get_unchecked_mut(base_v + dp_v) = best; }
                    d += 1;
                }
            }
        }
    }

    // --- Local begins + reporting for v=0 ---
    let v = 0usize;
    let mut vsc_root = ninf;
    let mut envi: i64 = (j0 + 1) as i64;
    let mut envj: i64 = (i0 - 1) as i64;
    let local_begin = cm.flags & (1 << 10) != 0; // CMH_LOCAL_BEGIN
    let jpx = (jmax[v] - jmin[v]) as i32;
    for jp_v_i in 0..=jpx.max(-1) {
        if jmax[v] < jmin[v] { break; }
        let jp_v = jp_v_i as usize;
        let j = jp_v as i32 + jmin[v];
        if local_begin {
            for y in 1..m {
                if cm.beginsc[y] > -9.999e35 && j >= jmin[y] && j <= jmax[y] {
                    let jp_y = (j - jmin[y]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    if dx >= dn {
                        let dp_v = (dn - hdmin[v][jp_v]) as usize;
                        let dp_y = (dn - hdmin[y][jp_y]) as usize;
                        let n = (dx - dn + 1) as usize;
                        let iv0 = roff_flat[roff_start[v] + jp_v] + dp_v;
                        let iy0 = roff_flat[roff_start[y] + jp_y] + dp_y;
                        // Same contiguous max-add as the emitting-state loop
                        // (bit-identical); reuse the AVX2 kernel.
                        #[cfg(target_arch = "x86_64")]
                        {
                            if cyk_use_avx2 {
                                unsafe { cyk_max_add_avx2(&mut alpha[..], iv0, iy0, n, cm.beginsc[y]) };
                            } else {
                                for k in 0..n {
                                    let sc = alpha[iy0 + k] + cm.beginsc[y];
                                    if sc > alpha[iv0 + k] { alpha[iv0 + k] = sc; }
                                }
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        for k in 0..n {
                            let sc = alpha[iy0 + k] + cm.beginsc[y];
                            if sc > alpha[iv0 + k] { alpha[iv0 + k] = sc; }
                        }
                    }
                }
            }
        }
        // vsc_root + envelope
        let dpx = hdmax[v][jp_v] - hdmin[v][jp_v];
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let sc = alpha[roff_flat[roff_start[v] + jp_v] + dp_v as usize];
            if sc > vsc_root {
                vsc_root = sc;
            }
        }
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let sc = alpha[roff_flat[roff_start[v] + jp_v] + dp_v as usize];
            if sc >= env_cutoff {
                let d = dp_v + hdmin[v][jp_v];
                let i = j - d + 1;
                if (i as i64) < envi { envi = i as i64; }
                if (j as i64) > envj { envj = j as i64; }
            }
        }
    }

    let ret_envi = if envi == (j0 + 1) as i64 { -1 } else { envi };
    let ret_envj = if envj == (i0 - 1) as i64 { -1 } else { envj };
    HB_ALPHA.with(|c| *c.borrow_mut() = std::mem::take(&mut alpha));
    (vsc_root, ret_envi, ret_envj)
}

// ===========================================================================
// (d3) F7: FastFInsideScanHB (cm_dpsearch.c:3833) — HMM-banded Inside scan.
//      Identical to F6 CYK except the DP combine steps use FLogsum (float, bits)
//      instead of ESL_MAX. The local-end reinit, emissions, clamps, local-begin
//      (still MAX+bestr), and vsc_root are all identical to F6.
// ===========================================================================

const FLOGSUM_TBL: usize = 23000; // infernal.h:178

/// C: flogsum_lookup (logsum.c:150) flogsum_lookup[i]=sreLOG2(1+sreEXP2(-i/INTSCALE)).
fn flogsum_table() -> &'static [f32; FLOGSUM_TBL] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Box<[f32; FLOGSUM_TBL]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0.0f32; FLOGSUM_TBL]);
        for i in 0..FLOGSUM_TBL {
            t[i] = sre_log2(1.0 + sre_exp2(-(i as f64) / 1000.0)) as f32;
        }
        t
    })
}

/// C: FLogsum (logsum.c:165). max + log2(1+2^(min-max)) via lookup; bits.
#[inline]
fn flogsum(s1: f32, s2: f32) -> f32 {
    let max = s1.max(s2);
    let min = s1.min(s2);
    if min == f32::NEG_INFINITY || (max - min) >= 23.0 {
        max
    } else {
        max + flogsum_table()[((max - min) * 1000.0) as usize]
    }
}

// Inside inner d-loop: alpha[iv0+k] = flogsum(alpha[iv0+k], alpha[iy0+k] + tsc).
// Vectorized flogsum, bit-identical to the scalar `flogsum`: max/min via SIMD,
// idx = (diff*1000) truncated (== `as usize`), table via gather, and the
// early-out (`min==-inf || diff>=23 -> max`) via a blend. Masked-out lanes clamp
// their gather index into range and are discarded by the blend, so no OOB read.
// Contiguous d (band pre-clamped), parent/child rows disjoint → no aliasing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn finside_logsum_add_avx2(
    alpha: &mut [f32], iv0: usize, iy0: usize, n: usize, tsc: f32, table: &[f32],
) {
    use std::arch::x86_64::*;
    let tv = _mm256_set1_ps(tsc);
    let scale = _mm256_set1_ps(1000.0);
    let thr = _mm256_set1_ps(23.0);
    let neginf = _mm256_set1_ps(f32::NEG_INFINITY);
    let maxidx = _mm256_set1_epi32((FLOGSUM_TBL - 1) as i32);
    let zero = _mm256_setzero_si256();
    let tptr = table.as_ptr();
    let mut k = 0usize;
    while k + 8 <= n {
        let a = _mm256_loadu_ps(alpha.as_ptr().add(iv0 + k));
        let b = _mm256_add_ps(_mm256_loadu_ps(alpha.as_ptr().add(iy0 + k)), tv);
        let mx = _mm256_max_ps(a, b);
        let mn = _mm256_min_ps(a, b);
        let diff = _mm256_sub_ps(mx, mn);
        let cond = _mm256_or_ps(
            _mm256_cmp_ps::<_CMP_EQ_OQ>(mn, neginf),
            _mm256_cmp_ps::<_CMP_GE_OQ>(diff, thr),
        );
        let mut idx = _mm256_cvttps_epi32(_mm256_mul_ps(diff, scale));
        idx = _mm256_max_epi32(_mm256_min_epi32(idx, maxidx), zero);
        let t = _mm256_i32gather_ps::<4>(tptr, idx);
        let out = _mm256_blendv_ps(_mm256_add_ps(mx, t), mx, cond);
        _mm256_storeu_ps(alpha.as_mut_ptr().add(iv0 + k), out);
        k += 8;
    }
    while k < n {
        let a_iy = *alpha.get_unchecked(iy0 + k);
        let av = alpha.get_unchecked_mut(iv0 + k);
        *av = flogsum(*av, a_iy + tsc);
        k += 1;
    }
}

/// C: FastFInsideScanHB (cm_dpsearch.c:3833). Returns (vsc_root, envi, envj, hits)
/// where `hits` = raw greedily-reported hits (i, j, null3-corrected score) BEFORE
/// overlap removal — the C `tmp_hitlist` contents (ReportHitsGreedily, cm_mx.c:7555).
/// `cutoff` is the reporting bit-score cutoff; `do_null3` enables the act/comp/null3
/// correction (TRUE in the default final stage).
pub fn fast_finside_scan_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    env_cutoff: f32,
    cutoff: f32,
    do_null3: bool,
) -> (f32, i64, i64, Vec<(i32, i32, f32, f32)>) {
    let m = cm.m as usize;
    let kp = crate::cm::ALPHABET_SIZE_P;
    #[cfg(target_arch = "x86_64")]
    let fi_use_avx2 = is_x86_feature_detected!("avx2");
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let w = (j0 - i0 + 1) as usize;

    let nj = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut roff_start: Vec<usize> = vec![0; m];
    let mut roff_flat: Vec<usize> = Vec::new();
    let mut ncells: usize = 0;
    for v in 0..m {
        let njv = nj(v);
        roff_start[v] = roff_flat.len();
        for jp in 0..njv {
            roff_flat.push(ncells);
            let width = hdmax[v][jp] - hdmin[v][jp] + 1;
            if width > 0 {
                ncells += width as usize;
            }
        }
    }
    let mut alpha = HB_ALPHA.with(|c| std::mem::take(&mut *c.borrow_mut()));
    alpha.clear();
    alpha.resize(ncells, CM_IMPOSSIBLE);
    let mut el_sca = vec![0.0f32; w + 1];
    for d in 0..=w {
        el_sca[d] = cm.el_selfsc * d as f32;
    }
    let ninf = CM_IMPOSSIBLE;

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        // Hoist v-indexed rows out of inner loops (byte-identical; checks stay on).
        let roff_v = &roff_flat[roff_start[v]..];
        let hdmin_v = &hdmin[v];
        let hdmax_v = &hdmax[v];
        let oesc_v = &cm.oesc[v];
        let tsc_v = &cm.tsc[v];

        if cm.endsc[v] > -9.999e35 {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut dp_v = 0usize;
                let mut d = hdmin_v[jp_v];
                while d <= hdmax_v[jp_v] {
                    let dp = (d - sd).max(0);
                    unsafe { *alpha.get_unchecked_mut(roff_v[jp_v] + dp_v) = el_sca[dp as usize] + cm.endsc[v]; }
                    dp_v += 1;
                    d += 1;
                }
            }
        }

        if stt == E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                unsafe { *alpha.get_unchecked_mut(roff_v[jp_v]) = 0.0; }
            }
        } else if stt == IL_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let base_v = roff_v[jp_v];
                let hdmin_vjp = hdmin_v[jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax_v[jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut cur = unsafe { *alpha.get_unchecked(base_v + dp_v) };
                    for &yoffset in &yvalid {
                        let y = cfirst + yoffset;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            cur = flogsum(cur, unsafe { *alpha.get_unchecked(roff_flat[roff_start[y] + jp_y_sdr] + dp_y_sd) } + tsc_v[yoffset]);
                        }
                    }
                    cur += oesc_v[dsq[i as usize] as usize];
                    if cur < ninf {
                        cur = ninf;
                    }
                    unsafe { *alpha.get_unchecked_mut(base_v + dp_v) = cur; }
                    d += 1;
                }
            }
        } else if stt == IR_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let base_v = roff_v[jp_v];
                let hdmin_vjp = hdmin_v[jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax_v[jp_v] {
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut cur = unsafe { *alpha.get_unchecked(base_v + dp_v) };
                    for &yoffset in &yvalid {
                        let y = cfirst + yoffset;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            cur = flogsum(cur, unsafe { *alpha.get_unchecked(roff_flat[roff_start[y] + jp_y_sdr] + dp_y_sd) } + tsc_v[yoffset]);
                        }
                    }
                    cur += oesc_v[dsq[j as usize] as usize];
                    if cur < ninf {
                        cur = ninf;
                    }
                    unsafe { *alpha.get_unchecked_mut(base_v + dp_v) = cur; }
                    d += 1;
                }
            }
        } else if stt != crate::constants::B_ST {
            for yoffset in 0..cnum {
                let y = cfirst + yoffset;
                let tsc = tsc_v[yoffset];
                let roff_y = &roff_flat[roff_start[y]..];
                let hdmin_y = &hdmin[y];
                let hdmax_y = &hdmax[y];
                let jn = jmin[v].max(jmin[y] + sdr);
                let jx = jmax[v].min(jmax[y] + sdr);
                let mut jp_v = (jn - jmin[v]) as i32;
                let mut jp_y_sdr = (jn - jmin[y] - sdr) as i32;
                let mut jj = jn;
                while jj <= jx {
                    let jpv = jp_v as usize;
                    let jpysdr = jp_y_sdr as usize;
                    let dn = hdmin_v[jpv].max(hdmin_y[jpysdr] + sd);
                    let dx = hdmax_v[jpv].min(hdmax_y[jpysdr] + sd);
                    let dp_v = (dn - hdmin_v[jpv]) as i32;
                    let dp_y_sd = (dn - hdmin_y[jpysdr] - sd) as i32;
                    let base_v = roff_v[jpv];
                    let base_y = roff_y[jpysdr];
                    if dx >= dn {
                        let n = (dx - dn + 1) as usize;
                        let iv0 = base_v + dp_v as usize;
                        let iy0 = base_y + dp_y_sd as usize;
                        #[cfg(target_arch = "x86_64")]
                        {
                            if fi_use_avx2 {
                                unsafe { finside_logsum_add_avx2(&mut alpha[..], iv0, iy0, n, tsc, &flogsum_table()[..]) };
                            } else {
                                for k in 0..n {
                                    let a_iy = unsafe { *alpha.get_unchecked(iy0 + k) };
                                    unsafe {
                                        let av = alpha.get_unchecked_mut(iv0 + k);
                                        *av = flogsum(*av, a_iy + tsc);
                                    }
                                }
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        for k in 0..n {
                            let a_iy = unsafe { *alpha.get_unchecked(iy0 + k) };
                            unsafe {
                                let av = alpha.get_unchecked_mut(iv0 + k);
                                *av = flogsum(*av, a_iy + tsc);
                            }
                        }
                    }
                    jp_v += 1;
                    jp_y_sdr += 1;
                    jj += 1;
                }
            }
            if stt == crate::constants::ML_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin_v[jp_v] + 1;
                    let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                    let base_v = roff_v[jp_v];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        unsafe { *alpha.get_unchecked_mut(base_v + dp_v as usize) += oesc_v[dsq[i as usize] as usize]; }
                        i -= 1;
                    }
                }
            } else if stt == crate::constants::MR_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                    let base_v = roff_v[jp_v];
                    let em = oesc_v[dsq[j as usize] as usize];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        unsafe { *alpha.get_unchecked_mut(base_v + dp_v as usize) += em; }
                    }
                }
            } else if stt == crate::constants::MP_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin_v[jp_v] + 1;
                    let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                    let base_v = roff_v[jp_v];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                        unsafe { *alpha.get_unchecked_mut(base_v + dp_v as usize) += oesc_v[idx]; }
                        i -= 1;
                    }
                }
            }
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let width = hdmax_v[jp_v] - hdmin_v[jp_v];
                let base_v = roff_v[jp_v];
                for dp_v in 0..=width.max(-1) {
                    if width < 0 { break; }
                    let idx = base_v + dp_v as usize;
                    if alpha[idx] < ninf {
                        alpha[idx] = ninf;
                    }
                }
            }
        } else {
            // B_st
            let y = cfirst;
            let z = cnum;
            let roff_y = &roff_flat[roff_start[y]..];
            let hdmin_y = &hdmin[y];
            let hdmax_y = &hdmax[y];
            let roff_z = &roff_flat[roff_start[z]..];
            let hdmin_z = &hdmin[z];
            let hdmax_z = &hdmax[z];
            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let hdmin_zjp = hdmin_z[jp_z];
                let kn0 = (j - jmax[y]).max(hdmin_zjp);
                let kn = kn0.max(0);
                let kx = jp_y.min(hdmax_z[jp_z]);
                let base_v = roff_v[jp_v];
                let base_z = roff_z[jp_z];
                let hdmin_vjp = hdmin_v[jp_v];
                let mut d = hdmin_vjp;
                while d <= hdmax_v[jp_v] {
                    let dp_v = (d - hdmin_vjp) as usize;
                    let mut cur = unsafe { *alpha.get_unchecked(base_v + dp_v) };
                    let mut k = kn;
                    while k <= kx {
                        let jp_y_mk = (jp_y - k) as usize;
                        if k >= d - hdmax_y[jp_y_mk] && k <= d - hdmin_y[jp_y_mk] {
                            let kp_z = (k - hdmin_zjp) as usize;
                            let dp_y = d - hdmin_y[jp_y_mk];
                            let iy = roff_y[jp_y_mk] + (dp_y - k) as usize;
                            let iz = base_z + kp_z;
                            cur = flogsum(cur, unsafe { *alpha.get_unchecked(iy) + *alpha.get_unchecked(iz) });
                        }
                        k += 1;
                    }
                    unsafe { *alpha.get_unchecked_mut(base_v + dp_v) = cur; }
                    d += 1;
                }
            }
        }
    }

    // Build act: act[jp][a] = count of residue a in dsq[i0..=i0+jp-1], jp=0..W.
    // (C prefills act with esl_abc_DCount; canonical residues only — degenerate deferred.)
    let mut act: Vec<[f64; 4]> = vec![[0.0; 4]; w + 1];
    if do_null3 {
        for jp in 1..=w {
            act[jp] = act[jp - 1];
            let res = dsq[(i0 as usize) + jp - 1] as usize;
            if res < 4 {
                act[jp][res] += 1.0;
            }
        }
    }

    // local begins + reporting (identical to F6: MAX + bestr; vsc_root = max)
    let v = 0usize;
    let mut vsc_root = ninf;
    let mut envi: i64 = (j0 + 1) as i64;
    let mut envj: i64 = (i0 - 1) as i64;
    let local_begin = cm.flags & (1 << 10) != 0;
    let jpx = (jmax[v] - jmin[v]) as i32;
    let mut hits: Vec<(i32, i32, f32, f32)> = Vec::new();
    let mut bestsc = vec![ninf; w + 1];
    let mut bestr = vec![0i32; w + 1];
    let mut comp = [0.0f32; 4];
    for jp_v_i in 0..=jpx.max(-1) {
        if jmax[v] < jmin[v] { break; }
        let jp_v = jp_v_i as usize;
        let j = jp_v as i32 + jmin[v];
        // reset bestr/bestsc for this j (over the valid d range)
        for d in 0..=w {
            bestr[d] = 0;
            bestsc[d] = ninf;
        }
        if local_begin {
            for y in 1..m {
                if cm.beginsc[y] > -9.999e35 && j >= jmin[y] && j <= jmax[y] {
                    let jp_y = (j - jmin[y]) as usize;
                    let dn = hdmin[v][jp_v].max(hdmin[y][jp_y]);
                    let dx = hdmax[v][jp_v].min(hdmax[y][jp_y]);
                    let mut dp_v = (dn - hdmin[v][jp_v]) as i32;
                    let mut dp_y = (dn - hdmin[y][jp_y]) as i32;
                    let mut d = dn;
                    while d <= dx {
                        let iv = roff_flat[roff_start[v] + jp_v] + dp_v as usize;
                        let iy = roff_flat[roff_start[y] + jp_y] + dp_y as usize;
                        let sc = alpha[iy] + cm.beginsc[y];
                        if sc > alpha[iv] {
                            alpha[iv] = sc;
                            bestr[d as usize] = y as i32;
                        }
                        dp_v += 1;
                        dp_y += 1;
                        d += 1;
                    }
                }
            }
        }
        let dpx = hdmax[v][jp_v] - hdmin[v][jp_v];
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let d = dp_v + hdmin[v][jp_v];
            let sc = alpha[roff_flat[roff_start[v] + jp_v] + dp_v as usize];
            bestsc[d as usize] = sc;
            if sc > vsc_root {
                vsc_root = sc;
            }
        }
        for dp_v in 0..=dpx.max(-1) {
            if dpx < 0 { break; }
            let sc = alpha[roff_flat[roff_start[v] + jp_v] + dp_v as usize];
            if sc >= env_cutoff {
                let d = dp_v + hdmin[v][jp_v];
                let i = j - d + 1;
                if (i as i64) < envi { envi = i as i64; }
                if (j as i64) > envj { envj = j as i64; }
            }
        }
        // ReportHitsGreedily (cm_mx.c:7555) — per-j greedy hit emission
        let dmin = hdmin[v][jp_v];
        let dmax = hdmax[v][jp_v];
        if dmin <= dmax {
            let mut max_sc_reported = ninf;
            let mut d = dmin.max(1);
            while d <= dmax {
                let i = j - d + 1;
                let mut hit_sc = bestsc[d as usize];
                if hit_sc > max_sc_reported && hit_sc >= cutoff && hit_sc > -9.999e35 {
                    let mut do_report = true;
                    // null3 bias correction; C stores this value as hit->bias for tblout
                    // (cm_pipeline.c: hit->bias = null3_correction) and subtracts it from score.
                    let mut hit_bias = 0.0f32;
                    if do_null3 {
                        let ip = (i - i0 + 1) as usize;
                        let jp = (j - i0 + 1) as usize;
                        for a in 0..4 {
                            comp[a] = (act[jp][a] - act[ip - 1][a]) as f32;
                        }
                        esl_vec_fnorm(&mut comp);
                        let null3_corr = score_correction_null3(&cm.null, &comp, d, cm.n3_omega as f32);
                        hit_sc -= null3_corr;
                        hit_bias = null3_corr;
                        do_report = hit_sc > max_sc_reported && hit_sc >= cutoff;
                    }
                    if do_report {
                        hits.push((i, j, hit_sc, hit_bias));
                        max_sc_reported = hit_sc;
                    }
                }
                d += 1;
            }
        }
        let _ = &bestr;
    }
    let ret_envi = if envi == (j0 + 1) as i64 { -1 } else { envi };
    let ret_envj = if envj == (i0 - 1) as i64 { -1 } else { envj };
    HB_ALPHA.with(|c| *c.borrow_mut() = std::mem::take(&mut alpha));
    (vsc_root, ret_envi, ret_envj, hits)
}

// ===========================================================================
// (d3a2) Per-hit mdl from/to: HMM-banded CYK alignment of the surviving hit
//   subsequence, then ParsetreeToCMBounds -> (cfrom_emit, cto_emit).
//   C reference: cm_pipeline.c pli_align_hit -> cp9_ShiftCMBands (hmmband.c) ->
//   DispatchSqAlignment (--acyk path == default optacc for mdl from/to) ->
//   cm_alignT_hb / cm_CYKInsideAlignHB (cm_dpalign.c) ->
//   ParsetreeToCMBounds (cm_parsetree.c) -> ad->cfrom_emit/cto_emit.
//   All pipeline hits are J (standard, non-truncated) mode.
// ===========================================================================

const USED_EL: i32 = -2; // shadow sentinel: local end (transition to EL, state cm.M)
const USED_LOCAL_BEGIN: i32 = -3; // shadow sentinel: local begin from root

/// C: cp9_ShiftCMBands (hmmband.c:4633), do_trunc=FALSE path. Shifts the i/j
/// bands in-place from the search (window) frame to the hit-local [1..Lp] frame
/// (ip = i-1), clamps to legal ranges, then recomputes the d bands. `i`,`j` are
/// the hit's window-frame coords (same frame the search bands were computed in).
pub fn shift_cm_bands(cm: &CM, cp9b: &mut CP9Bands, i: i32, j: i32) {
    let ip = i - 1;
    let lp = j - i + 1;
    let m = cm.m as usize;
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        if cp9b.imin[v] > 0 {
            // state currently reachable
            let min_i = 1;
            let max_i = lp + 1 - sd;
            let min_j = sd;
            let max_j = lp.max(min_j);
            cp9b.imin[v] = (cp9b.imin[v] - ip).max(min_i);
            cp9b.imax[v] = (cp9b.imax[v] - ip).min(max_i);
            cp9b.jmin[v] = (cp9b.jmin[v] - ip).max(min_j);
            cp9b.jmax[v] = (cp9b.jmax[v] - ip).min(max_j);
            if cp9b.imax[v] < min_i
                || cp9b.jmax[v] < min_j
                || cp9b.imin[v] > max_i
                || cp9b.jmin[v] > max_j
            {
                cp9b.imin[v] = -1;
                cp9b.jmin[v] = -1;
                cp9b.imax[v] = -2;
                cp9b.jmax[v] = -2;
            }
        }
    }
    ij2d_bands(cm, lp, cp9b, false);
}

/// HMM-banded CYK alignment (port of cm_CYKInsideAlignHB, cm_dpalign.c:1102) with
/// shadow-matrix traceback (cm_alignT_hb CYK path) followed by ParsetreeToCMBounds
/// (cm_parsetree.c, J mode). `cp9b` must already be shifted to the [1..lp] frame
/// (see `shift_cm_bands`). `dsq` must be indexed so dsq[1..=lp] are the hit
/// residues (pass `&wdsq[(ws_loc - 1)..]`). `emap` is the CM emit map (build once
/// via `create_emit_map`). Returns (cfrom_emit, cto_emit) = tblout mdl from/to.
/// Falls back to (1, clen) if the shifted bands cannot admit a full alignment.
pub fn cyk_align_hb_cmbounds(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[u8],
    lp: i32,
    emap: &crate::CMEmitMap,
) -> (i32, i32) {
    let m = cm.m as usize;
    let kp = crate::cm::ALPHABET_SIZE_P;
    let jmin = &cp9b.jmin;
    let jmax = &cp9b.jmax;
    let hdmin = &cp9b.hdmin;
    let hdmax = &cp9b.hdmax;
    let ninf = CM_IMPOSSIBLE;
    let local_begin = cm.flags & CMH_LOCAL_BEGIN != 0;

    // Validate a full alignment to ROOT_S (v==0) is admitted by the bands.
    if jmin[0] > lp || jmax[0] < lp {
        return (1, cm.clen);
    }
    let jp_0 = (lp - jmin[0]) as usize;
    if hdmin[0][jp_0] > lp || hdmax[0][jp_0] < lp {
        return (1, cm.clen);
    }
    let lp_0 = (lp - hdmin[0][jp_0]) as usize;

    // Allocate banded matrices: alpha[v][jp][dp], yshadow, kshadow.
    let njv = |v: usize| -> usize {
        if jmax[v] >= jmin[v] { (jmax[v] - jmin[v] + 1) as usize } else { 0 }
    };
    let mut alpha: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
    let mut yshadow: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    let mut kshadow: Vec<Vec<Vec<i32>>> = Vec::with_capacity(m);
    for v in 0..m {
        let nj = njv(v);
        let mut av = Vec::with_capacity(nj);
        let mut ys = Vec::with_capacity(nj);
        let mut ks = Vec::with_capacity(nj);
        for jp in 0..nj {
            let width = (hdmax[v][jp] - hdmin[v][jp] + 1).max(0) as usize;
            av.push(vec![ninf; width]);
            ys.push(vec![USED_EL; width]); // init all shadow cells to local-end
            ks.push(vec![0i32; width]);
        }
        alpha.push(av);
        yshadow.push(ys);
        kshadow.push(ks);
    }

    let el_selfsc = cm.el_selfsc;
    let mut b: i32 = -1;
    let mut bsc: f32 = ninf;

    for v in (0..m).rev() {
        let stt = cm.sttype[v] as i32;
        let sd = state_delta(stt);
        let sdr = state_right_delta(stt);
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;
        let tsc_v = &cm.tsc[v];
        let oesc_v = &cm.oesc[v];

        // re-initialize deck v if we can do a local end from v (transition to EL)
        if cm.endsc[v] > -9.999e35 {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    if d >= sd {
                        let dp_v = (d - hdmin[v][jp_v]) as usize;
                        alpha[v][jp_v][dp_v] = el_selfsc * (d - sd) as f32 + cm.endsc[v];
                    }
                    d += 1;
                }
            }
        }

        if stt == crate::constants::E_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                alpha[v][jp_v][0] = 0.0;
            }
        } else if stt == crate::constants::IL_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let i = j - d + 1;
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    for &yoffset in &yvalid {
                        let y = cfirst + yoffset;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            let sc = alpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset];
                            if sc > alpha[v][jp_v][dp_v] {
                                alpha[v][jp_v][dp_v] = sc;
                                yshadow[v][jp_v][dp_v] = yoffset as i32;
                            }
                        }
                    }
                    alpha[v][jp_v][dp_v] += oesc_v[dsq[i as usize] as usize];
                    if alpha[v][jp_v][dp_v] < ninf {
                        alpha[v][jp_v][dp_v] = ninf;
                    }
                    d += 1;
                }
            }
        } else if stt == crate::constants::IR_ST {
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let j_sdr = j - sdr;
                let mut yvalid: Vec<usize> = Vec::new();
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    if j_sdr >= jmin[y] && j_sdr <= jmax[y] {
                        yvalid.push(yoffset);
                    }
                }
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    for &yoffset in &yvalid {
                        let y = cfirst + yoffset;
                        let jp_y_sdr = (j - jmin[y] - sdr) as usize;
                        if (d - sd) >= hdmin[y][jp_y_sdr] && (d - sd) <= hdmax[y][jp_y_sdr] {
                            let dp_y_sd = (d - sd - hdmin[y][jp_y_sdr]) as usize;
                            let sc = alpha[y][jp_y_sdr][dp_y_sd] + tsc_v[yoffset];
                            if sc > alpha[v][jp_v][dp_v] {
                                alpha[v][jp_v][dp_v] = sc;
                                yshadow[v][jp_v][dp_v] = yoffset as i32;
                            }
                        }
                    }
                    alpha[v][jp_v][dp_v] += oesc_v[dsq[j as usize] as usize];
                    if alpha[v][jp_v][dp_v] < ninf {
                        alpha[v][jp_v][dp_v] = ninf;
                    }
                    d += 1;
                }
            }
        } else if stt != crate::constants::B_ST {
            // ML, MP, MR, D, S: children independent; for y { for j { for d } }
            for yoffset in 0..cnum {
                let y = cfirst + yoffset;
                let tsc = tsc_v[yoffset];
                let jn = jmin[v].max(jmin[y] + sdr);
                let jx = jmax[v].min(jmax[y] + sdr);
                let mut jp_v = jn - jmin[v];
                let mut jp_y_sdr = jn - jmin[y] - sdr;
                let mut jj = jn;
                while jj <= jx {
                    let jpv = jp_v as usize;
                    let jpysdr = jp_y_sdr as usize;
                    let dn = hdmin[v][jpv].max(hdmin[y][jpysdr] + sd);
                    let dx = hdmax[v][jpv].min(hdmax[y][jpysdr] + sd);
                    let mut dp_v = dn - hdmin[v][jpv];
                    let mut dp_y_sd = dn - hdmin[y][jpysdr] - sd;
                    let mut d = dn;
                    while d <= dx {
                        let sc = alpha[y][jpysdr][dp_y_sd as usize] + tsc;
                        if sc > alpha[v][jpv][dp_v as usize] {
                            alpha[v][jpv][dp_v as usize] = sc;
                            yshadow[v][jpv][dp_v as usize] = yoffset as i32;
                        }
                        dp_v += 1;
                        dp_y_sd += 1;
                        d += 1;
                    }
                    jp_v += 1;
                    jp_y_sdr += 1;
                    jj += 1;
                }
            }
            // add emission score, if any
            if stt == crate::constants::ML_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin[v][jp_v] + 1;
                    let width = hdmax[v][jp_v] - hdmin[v][jp_v];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        alpha[v][jp_v][dp_v as usize] += oesc_v[dsq[i as usize] as usize];
                        i -= 1;
                    }
                }
            } else if stt == crate::constants::MR_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let width = hdmax[v][jp_v] - hdmin[v][jp_v];
                    let em = oesc_v[dsq[j as usize] as usize];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        alpha[v][jp_v][dp_v as usize] += em;
                    }
                }
            } else if stt == crate::constants::MP_ST {
                for j in jmin[v]..=jmax[v] {
                    let jp_v = (j - jmin[v]) as usize;
                    let mut i = j - hdmin[v][jp_v] + 1;
                    let width = hdmax[v][jp_v] - hdmin[v][jp_v];
                    for dp_v in 0..=width.max(-1) {
                        if width < 0 { break; }
                        let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                        alpha[v][jp_v][dp_v as usize] += oesc_v[idx];
                        i -= 1;
                    }
                }
            }
            // ensure all cells >= IMPOSSIBLE
            for j in jmin[v]..=jmax[v] {
                let jp_v = (j - jmin[v]) as usize;
                let width = hdmax[v][jp_v] - hdmin[v][jp_v];
                for dp_v in 0..=width.max(-1) {
                    if width < 0 { break; }
                    if alpha[v][jp_v][dp_v as usize] < ninf {
                        alpha[v][jp_v][dp_v as usize] = ninf;
                    }
                }
            }
        } else {
            // B_st: cfirst=left child (y), cnum=right child (z)
            let y = cfirst;
            let z = cnum;
            let jn = jmin[v].max(jmin[z]);
            let jx = jmax[v].min(jmax[z]);
            for j in jn..=jx {
                let jp_v = (j - jmin[v]) as usize;
                let jp_y = j - jmin[y];
                let jp_z = (j - jmin[z]) as usize;
                let hdmin_zjp = hdmin[z][jp_z];
                let kn = (j - jmax[y]).max(hdmin_zjp).max(0);
                let kx = jp_y.min(hdmax[z][jp_z]);
                let mut d = hdmin[v][jp_v];
                while d <= hdmax[v][jp_v] {
                    let dp_v = (d - hdmin[v][jp_v]) as usize;
                    let mut k = kn;
                    while k <= kx {
                        let jp_y_mk = (jp_y - k) as usize;
                        if k >= d - hdmax[y][jp_y_mk] && k <= d - hdmin[y][jp_y_mk] {
                            let kp_z = (k - hdmin_zjp) as usize;
                            let dp_y = d - hdmin[y][jp_y_mk];
                            let sc = alpha[y][jp_y_mk][(dp_y - k) as usize]
                                + alpha[z][jp_z][kp_z];
                            if sc > alpha[v][jp_v][dp_v] {
                                alpha[v][jp_v][dp_v] = sc;
                                kshadow[v][jp_v][dp_v] = k;
                            }
                        }
                        k += 1;
                    }
                    d += 1;
                }
            }
        }

        // allow local begins, if nec
        if local_begin && lp >= jmin[v] && lp <= jmax[v] {
            let jp_v = (lp - jmin[v]) as usize;
            if lp >= hdmin[v][jp_v] && lp <= hdmax[v][jp_v] {
                let lp_v = (lp - hdmin[v][jp_v]) as usize;
                if cm.beginsc[v] > -9.999e35 && alpha[v][jp_v][lp_v] + cm.beginsc[v] > bsc {
                    b = v as i32;
                    bsc = alpha[v][jp_v][lp_v] + cm.beginsc[v];
                }
            }
        }
    }

    // store optimal local begin as overall score if it wins
    if bsc > -9.999e35 && bsc > alpha[0][jp_0][lp_0] {
        alpha[0][jp_0][lp_0] = bsc;
        yshadow[0][jp_0][lp_0] = USED_LOCAL_BEGIN;
    }

    // ---- traceback (CYK path of cm_alignT_hb) ----
    // Collect visited states in insertion order (root first). Order preserved so
    // the EL (v==cm.M) case can read its predecessor.
    let m_i = cm.m as i32;
    let mut states: Vec<i32> = vec![0];
    let mut pda: Vec<(i32, i32, i32)> = Vec::new(); // (saved j, subseq len k, B state)
    let mut v: i32 = 0;
    let mut i: i32 = 1;
    let mut j: i32 = lp;
    let mut d: i32 = lp;
    loop {
        if v == m_i || cm.sttype[v as usize] as i32 == crate::constants::E_ST {
            // E or EL: swing to a pending right subtree, or finish
            match pda.pop() {
                None => break,
                Some((pj, pk, bstate)) => {
                    j = pj;
                    d = pk;
                    v = cm.cnum[bstate as usize]; // right child S
                    i = j - d + 1;
                    states.push(v);
                    continue;
                }
            }
        }
        let vu = v as usize;
        let jp_v = (j - jmin[vu]) as usize;
        let dp_v = (d - hdmin[vu][jp_v]) as usize;
        if cm.sttype[vu] as i32 == crate::constants::B_ST {
            let k = kshadow[vu][jp_v][dp_v];
            pda.push((j, k, v));
            j -= k;
            d -= k;
            i = j - d + 1;
            let yy = cm.cfirst[vu];
            states.push(yy);
            v = yy;
            continue;
        }
        let yoffset = yshadow[vu][jp_v][dp_v];
        match cm.sttype[vu] as i32 {
            x if x == crate::constants::MP_ST => { i += 1; j -= 1; }
            x if x == crate::constants::ML_ST => { i += 1; }
            x if x == crate::constants::MR_ST => { j -= 1; }
            x if x == crate::constants::IL_ST => { i += 1; }
            x if x == crate::constants::IR_ST => { j -= 1; }
            _ => {} // D_st, S_st
        }
        d = j - i + 1;
        if yoffset == USED_EL {
            states.push(m_i);
            v = m_i;
        } else if yoffset == USED_LOCAL_BEGIN {
            states.push(b);
            v = b;
        } else {
            let yy = cm.cfirst[vu] + yoffset;
            states.push(yy);
            v = yy;
        }
    }

    // ---- ParsetreeToCMBounds (J mode) -> (cfrom_emit, cto_emit) ----
    let clen = cm.clen;
    let mut cfrom = clen + 1;
    let mut cto = 0;
    let mut insert_sd = 0i32; // persists across nodes; not reset for the EL case
    let mp_nd = crate::constants::MATP_ND;
    let ml_nd = crate::constants::MATL_ND;
    let mr_nd = crate::constants::MATR_ND;
    for idx in 0..states.len() {
        let vv = states[idx];
        let (lpos, rpos, is_left, is_right);
        if vv != m_i {
            let nd = cm.ndidx[vv as usize] as usize;
            let ndt = cm.ndtype[nd] as i32;
            lpos = if ndt == mp_nd || ndt == ml_nd { emap.lpos[nd] } else { emap.lpos[nd] + 1 };
            rpos = if ndt == mp_nd || ndt == mr_nd { emap.rpos[nd] } else { emap.rpos[nd] - 1 };
            let stt = cm.sttype[vv as usize] as i32;
            if stt == crate::constants::IL_ST {
                is_left = true; is_right = false; insert_sd = 1;
            } else if stt == crate::constants::IR_ST {
                is_left = false; is_right = true; insert_sd = 1;
            } else if ndt == mp_nd {
                is_left = true; is_right = true; insert_sd = 0;
            } else if ndt == ml_nd {
                is_left = true; is_right = false; insert_sd = 0;
            } else if ndt == mr_nd {
                is_left = false; is_right = true; insert_sd = 0;
            } else {
                is_left = false; is_right = false; insert_sd = 0;
            }
        } else {
            // v == cm.M (EL): use node that EL replaced (1 + node of prev state).
            // insert_sd retains its previous value (matches C).
            let prv = states[idx - 1] as usize;
            let nd = 1 + cm.ndidx[prv] as usize;
            let ndt = cm.ndtype[nd] as i32;
            lpos = if ndt == mp_nd || ndt == ml_nd { emap.lpos[nd] } else { emap.lpos[nd] + 1 };
            rpos = if ndt == mp_nd || ndt == mr_nd { emap.rpos[nd] } else { emap.rpos[nd] - 1 };
            is_left = true; is_right = true;
        }
        if is_left {
            cfrom = cfrom.min(lpos + insert_sd);
            cto = cto.max(lpos);
        }
        if is_right {
            cfrom = cfrom.min(rpos + insert_sd);
            cto = cto.max(rpos);
        }
    }
    (cfrom, cto)
}

// ===========================================================================
// (d3b) Scoring transforms for final hit reporting: null3 correction + E-value.
// ===========================================================================

/// C: ScoreCorrectionNull3 (cm_parsetree.c:2352). Given the hit's residue
/// composition <comp> (freqs, sum≈1), length <len>, the build null model
/// <null0>, and prior <omega>, returns the log2-odds correction that the caller
/// SUBTRACTS from the hit's bit score. K=4 (RNA canonical).
pub fn score_correction_null3(null0: &[f32], comp: &[f32], len: i32, omega: f32) -> f32 {
    let k = ALPHABET_SIZE;
    // C computes each term in double (sreLOG2 is double; comp/len promote) and
    // accumulates into a float `score` (score += <double>), rounding per step.
    let mut score: f32 = 0.0;
    for a in 0..k {
        let term = sre_log2((comp[a] / null0[a]) as f64) * comp[a] as f64 * len as f64;
        score = (score as f64 + term) as f32;
    }
    score = (score as f64 + sre_log2(omega as f64)) as f32;
    // LogSum2(0., score) == FLogsum(0, score)
    flogsum(0.0, score)
}

/// C: esl_exp_surv (esl_exponential.c). Survival P(X>x) for an exponential tail.
pub fn esl_exp_surv(x: f64, mu: f64, lambda: f64) -> f64 {
    if x < mu {
        1.0
    } else {
        (-lambda * (x - mu)).exp()
    }
}

/// C: cm_tophits_SortForOverlapRemoval + RemoveOrMarkOverlaps(do_remove=TRUE) +
/// remove_or_mark_overlaps_one_seq_memeff (cm_tophits.c), single model/seq/forward
/// strand. Sorts hits by score desc then start asc, then greedily marks every
/// lower-scoring hit that overlaps a kept higher-scoring hit as removed. Returns
/// the survivors in sorted order. Input/output tuples are (start, stop, score).
pub fn remove_overlaps_greedy(mut hits: Vec<(i32, i32, f32, f32)>) -> Vec<(i32, i32, f32, f32)> {
    // sort: 4th key score high->low, 5th key start low->high (single seq/strand)
    hits.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let n = hits.len();
    let mut removed = vec![false; n];
    for i in 0..n {
        if removed[i] {
            continue;
        }
        let (si, ei, _, _) = hits[i]; // start_i, stop_i
        for j in (i + 1)..n {
            if removed[j] {
                continue;
            }
            let (sj, ej, _, _) = hits[j];
            // forward overlap: NOT(stop_j < start_i) AND NOT(stop_i < start_j)
            if !(ej < si) && !(ei < sj) {
                removed[j] = true;
            }
        }
    }
    hits.into_iter()
        .zip(removed)
        .filter(|(_, r)| !*r)
        .map(|(h, _)| h)
        .collect()
}
