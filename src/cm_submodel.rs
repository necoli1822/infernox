//! Sub-CM alignment support (`cmalign --sub`).
//!
//! Faithful port of C Infernal `cm_submodel.c` (+ the sub-alignment prep in
//! `cm_alndata.c`). For each target sequence, the CP9 HMM predicts the
//! consensus start/end columns the sequence spans; a temporary "sub CM"
//! covering only those columns is built from the (global) original CM, the
//! sequence is aligned to the sub CM, and the sub-CM parsetree is mapped back
//! to full-model coordinates for output.
//!
//! Staged port. This module currently implements Stage 2: the CP9 posterior +
//! CP9NodeForPosn machinery that decides the sub-CM's consensus start/end
//! columns (cfrom/cto) for each sequence.

use crate::cm::CM;
use crate::cm_consensus::create_cm_consensus_full;
use crate::cm_modelmaker::consensus_modelmaker;
use crate::cm_rebalance::cm_rebalance;
use crate::cm::ALPHABET_SIZE;
use crate::cm_modelmaker::cm_find_and_detach_dual_inserts;
use crate::constants::{
    B_ST, BEGL_ND, BEGR_ND, BIF_ND, D_ST, E_ST, END_ND, IL_ST, IR_ST, MATL_ML, MATL_ND, MATP_MP,
    MATP_ND, MATR_MR, MATR_ND, ML_ST, MP_ST, MR_ST, ROOT_ND, S_ST,
};
use crate::parsetree::Parsetree;
use crate::cp9::{
    cm_create_transition_map, cm_expected_state_occupancy, cp9_backward, cp9_forward,
    cp9_map_cm2hmm, ilogsum, Cp9PostMx, CP9Map, CP9,
};
use crate::easel::alphabet::EslAlphabet;

/// C hard-coded transition map type (cp9): tmap[stid_x][ndtype][stid_v] -> yoffset.
type Tmap = Vec<Vec<Vec<i8>>>;

/// C `#define INFTY 987654321` (infernal.h:157) — the integer-DP -inf sentinel.
const INFTY: i32 = crate::cp9::INFTY;

// =============================================================================
// Stage 2: CP9 posterior + start/end consensus-column prediction.
// =============================================================================

/// C: cp9_Posterior (hmmband.c:1148). Combine CP9 Forward (`fmx`) and Backward
/// (`bmx`) integer log-odds matrices into a posterior matrix `mx`, where
/// `mx.mmx[ip][k]` is the log-odds score that residue ip was emitted by node k's
/// match state (and likewise imx/dmx). `did_fwd_scan` selects the normalization
/// constant `sc` (summed over all end positions for a scanning Forward, else the
/// single overall score `bmx.mmx[0][0]`).
pub fn cp9_posterior(
    dsq: &[u8],
    i0: usize,
    j0: usize,
    hmm: &CP9,
    fmx: &crate::cp9::CP9Mx,
    bmx: &crate::cp9::CP9Mx,
    did_fwd_scan: bool,
) -> Cp9PostMx {
    let m = hmm.m as usize;
    let l = j0 - i0 + 1;

    // sc = summed log prob of all parses (hmmband.c:1160-1171)
    let sc: i32 = if did_fwd_scan {
        let mut s = -INFTY;
        for ip in 0..=l {
            s = ilogsum(s, bmx.mmx[ip][0]);
        }
        s
    } else {
        bmx.mmx[0][0]
    };

    let mut mmx = vec![vec![-INFTY; m + 1]; l + 1];
    let mut imx = vec![vec![-INFTY; m + 1]; l + 1];
    let mut dmx = vec![vec![-INFTY; m + 1]; l + 1];

    // Boundary conditions (hmmband.c:1173-1183)
    mmx[0][0] = fmx.mmx[0][0] + bmx.mmx[0][0] - sc;
    imx[0][0] = -INFTY; // need seq to get here
    dmx[0][0] = -INFTY; // D_0 does not exist
    for k in 1..=m {
        mmx[0][k] = -INFTY; // need seq to get here
        imx[0][k] = -INFTY; // need seq to get here
        dmx[0][k] = fmx.dmx[0][k] + bmx.dmx[0][k] - sc;
    }

    // hmmband.c:1185-1204. Note the emission score msc/isc counted in BOTH fmx
    // and bmx is subtracted out. Rust msc/isc are indexed [k][sym] (transposed
    // from C's [sym][k]); C's `hmm->msc[dsq[i]][k]` == `hmm.msc[k][dsq[i]]`.
    for ip in 1..=l {
        let i = i0 + ip - 1; // actual index in dsq, i0..j0
        let di = dsq[i] as usize;
        mmx[ip][0] = -INFTY; // M_0 does not emit
        imx[ip][0] = fmx.imx[ip][0] + bmx.imx[ip][0] - hmm.isc[0][di] - sc;
        dmx[ip][0] = -INFTY; // D_0 does not exist
        for k in 1..=m {
            mmx[ip][k] = (fmx.mmx[ip][k] + bmx.mmx[ip][k] - hmm.msc[k][di] - sc).max(-INFTY);
            imx[ip][k] = (fmx.imx[ip][k] + bmx.imx[ip][k] - hmm.isc[k][di] - sc).max(-INFTY);
            dmx[ip][k] = (fmx.dmx[ip][k] + bmx.dmx[ip][k] - sc).max(-INFTY);
        }
    }

    Cp9PostMx { mmx, imx, dmx }
}

/// C: cp9_Seq2Posteriors (hmmband.c:513). Run CP9 Forward + Backward over the
/// full sequence (non-scanning, alignment mode) and return the posterior matrix.
/// The caller supplies the (global) `hmm` (C uses `cm->cp9`).
pub fn cp9_seq2posteriors(hmm: &CP9, dsq: &[u8], i0: usize, j0: usize) -> Cp9PostMx {
    // C cp9_Forward/cp9_Backward: do_scan=FALSE, doing_align=TRUE (hmmband.c:531,540).
    let (_fsc, _i, fmx, _sca) = cp9_forward(hmm, dsq, i0, j0, false, true);
    let (_bsc, _j, bmx, _scb) = cp9_backward(hmm, dsq, i0, j0, false, true);
    // cp9_Posterior with did_fwd_scan=FALSE (hmmband.c:551).
    cp9_posterior(dsq, i0, j0, hmm, &fmx, &bmx, false)
}

/// C: CP9NodeForPosn (cm_submodel.c:844). Return the HMM node (and state type,
/// 0=match / 1=insert) most likely to have emitted position `x` of the target,
/// per the posterior matrix `post`. `is_start`/`pmass` are vestigial in C (only
/// used by a disabled debug print), so they are omitted here. `post` is indexed
/// by relative position; with i0=1 the row index equals the absolute position x.
pub fn cp9_node_for_posn(hmm: &CP9, x: usize, post: &Cp9PostMx) -> (i32, i32) {
    // Initialize from node 0 (cm_submodel.c:865-876).
    let mut max_k: i32 = 0;
    let (mut max_sc, mut max_type) = if post.mmx[x][0] > post.imx[x][0] {
        (post.mmx[x][0], 0) // match
    } else {
        (post.imx[x][0], 1) // insert
    };
    // Move left to right through HMM nodes (cm_submodel.c:877-891).
    for k in 1..=hmm.m as usize {
        if post.mmx[x][k] > max_sc {
            max_k = k as i32;
            max_sc = post.mmx[x][k];
            max_type = 0; // match
        }
        if post.imx[x][k] > max_sc {
            max_k = k as i32;
            max_sc = post.imx[x][k];
            max_type = 1; // insert
        }
    }
    (max_k, max_type)
}

/// C: sub_alignment_prep() step 1 (cm_alndata.c:122-140). Predict the sub-CM's
/// consensus start (`spos`) and end (`epos`) columns for a target sequence, from
/// the global CP9 posterior. Returns (spos, epos) = (sstruct, estruct) as passed
/// to build_sub_cm.
pub fn predict_sub_cm_columns(hmm: &CP9, dsq: &[u8], l: i32) -> (i32, i32) {
    let l = l as usize;
    // cp9_Seq2Posteriors(orig_cm, ..., dsq, 1, sq->L, 0) (cm_alndata.c:123).
    let post = cp9_seq2posteriors(hmm, dsq, 1, l);
    // CP9NodeForPosn(cp9, 1, L, 1, ..., is_start=TRUE)  -> spos
    // CP9NodeForPosn(cp9, 1, L, L, ..., is_start=FALSE) -> epos
    let (mut spos, spos_state) = cp9_node_for_posn(hmm, 1, &post);
    let (mut epos, epos_state) = cp9_node_for_posn(hmm, l, &post);
    // Special cases (cm_alndata.c:126-140).
    // If the most likely state to have emitted the first/last residue is the
    // insert state in node 0, model from consensus column 1.
    if spos == 0 && spos_state == 1 {
        spos = 1;
    }
    if epos == 0 && epos_state == 1 {
        epos = 1;
    }
    // If the end node comes before-or-equals the start node, the HMM alignment
    // is unreliable: default to the full CM (spos=1, epos=cp9->M).
    if epos <= spos {
        spos = 1;
        epos = hmm.m;
    }
    (spos, epos)
}

// =============================================================================
// Stage 3a: sub-CM topology construction (the structural skeleton, no
// probabilities yet). C build_sub_cm (cm_submodel.c:563) up through CMRebalance.
// =============================================================================

/// C: build_sub_cm (cm_submodel.c:598-676), the topology portion. Given the
/// (configured, global) original CM and the sub-CM consensus column bounds
/// [spos..epos] (== [sstruct..estruct] in the sub_alignment_prep call path),
/// build the count-zeroed sub-CM skeleton: derive the sub consensus structure
/// string from the original CM's consensus ct, run ConsensusModelmaker, then
/// CMRebalance. Probabilities are filled in later stages.
pub fn build_sub_cm_topology(orig_cm: &CM, spos: i32, epos: i32) -> CM {
    // sstruct/estruct == spos/epos in the sub_alignment_prep call path
    // (cm_alndata.c:143-145).
    let sstruct = spos;
    let estruct = epos;

    // Consensus sequence/structure of the original CM (C CreateCMConsensus,
    // cm_submodel.c:599). con.ct uses -1 for unpaired (0..clen-1).
    let con = create_cm_consensus_full(orig_cm)
        .expect("build_sub_cm_topology: CreateCMConsensus failed (CM needs bit scores)");

    // Fill sub_ct for [spos..epos] (cm_submodel.c:620-651). sub_ct is indexed
    // 0..(epos-spos); its VALUES stay in the original coordinate system, matching
    // C. NOTE: with sstruct==spos and estruct==epos the structure-removal loop
    // (second loop) never fires, so no out-of-bounds occurs.
    let n = (epos - spos + 1) as usize;
    let mut sub_ct = vec![0i32; n];
    // First: copy ct for the model boundaries (cm_submodel.c:626-635).
    let mut cpos = spos - 1;
    while cpos < epos {
        let sub_cpos = (cpos - (spos - 1)) as usize;
        let cc = con.ct[cpos as usize];
        if cc != -1 && (cc < (spos - 1) || cc >= epos) {
            sub_ct[sub_cpos] = -1;
        } else {
            sub_ct[sub_cpos] = cc;
        }
        cpos += 1;
    }
    // Second: remove structure outside structural boundaries (cm_submodel.c:637-651).
    // (Dead when sstruct==spos and estruct==epos, but transcribed faithfully.)
    let mut cpos = spos - 1;
    while cpos < epos {
        let sub_cpos = (cpos - (spos - 1)) as usize;
        if (cpos + 1) < sstruct || (cpos + 1) > estruct {
            if sub_ct[sub_cpos] != -1 {
                let p = sub_ct[sub_cpos];
                sub_ct[p as usize] = -1;
            }
            sub_ct[sub_cpos] = -1;
        }
        cpos += 1;
    }

    // Build the sub consensus structure string (angle-bracket/dot only)
    // (cm_submodel.c:658-670).
    let mut sub_cstr = vec![0u8; n];
    let mut cpos = spos - 1;
    while cpos < epos {
        let sub_cpos = (cpos - (spos - 1)) as usize;
        if sub_ct[sub_cpos] == -1 {
            sub_cstr[sub_cpos] = b'.';
        } else if sub_ct[sub_cpos] > cpos {
            sub_cstr[sub_cpos] = b'<';
        } else if sub_ct[sub_cpos] < cpos {
            sub_cstr[sub_cpos] = b'>';
        } else {
            panic!("build_sub_cm_topology: weird ct self-pair at cpos {}", cpos);
        }
        cpos += 1;
    }

    // ConsensusModelmaker (building_sub_model=TRUE) -> guide tree -> sub_cm
    // (cm_submodel.c:673). Then CMRebalance (cm_submodel.c:681-684).
    let abc = EslAlphabet::rna();
    let (sub_cm, _mtr) = consensus_modelmaker(&abc, &sub_cstr, epos - spos + 1, true);
    cm_rebalance(&sub_cm)
}

// =============================================================================
// Stage 3b: orig <-> sub state/node maps. C AllocSubMap (cm_submodel.c:83) +
// map_orig2sub_cm (:228) + map_orig2sub_cm_helper (:419) +
// cm2sub_cm_check_id_next_node (:2525).
// =============================================================================

/// C stid access for the oversized (0..=M) arrays: index M is the EL state.
#[inline]
fn stid_at(cm: &CM, v: usize) -> i32 {
    if v < cm.stid.len() { cm.stid[v] as i32 } else { -1 }
}
/// C sttype access; index M (== cm.m) is the EL state (sttype EL_st).
#[inline]
fn sttype_at(cm: &CM, v: usize) -> i32 {
    if v < cm.sttype.len() { cm.sttype[v] as i32 } else { crate::constants::EL_ST }
}

/// C: CMSubMap_t (infernal.h:898). Map of a template CM to a sub CM and back.
pub struct CMSubMap {
    pub spos: i32,
    pub epos: i32,
    pub sstruct: i32,
    pub estruct: i32,
    /// s2o_smap[v][0..1]: orig_cm state(s) that sub_cm state v maps to. v=0..sub_M.
    pub s2o_smap: Vec<[i32; 2]>,
    /// o2s_smap[v][0..1]: sub_cm state(s) that orig_cm state v maps to. v=0..orig_M.
    pub o2s_smap: Vec<[i32; 2]>,
    /// s2o_id[v]: TRUE if sub_cm state v maps identically to an orig_cm state.
    pub s2o_id: Vec<bool>,
    pub sub_clen: i32,
    pub orig_clen: i32,
    pub sub_m: i32,
    pub orig_m: i32,
}

/// C: AllocSubMap (cm_submodel.c:83). Allocate + initialize the sub map. Arrays
/// sized M+1 (indices 0..=M, the extra slot being the EL state), maps init to -1,
/// s2o_id init to FALSE. sstruct/estruct == spos/epos in the call path.
pub fn alloc_sub_map(sub_cm: &CM, orig_cm: &CM, sstruct: i32, estruct: i32) -> CMSubMap {
    let sub_m = sub_cm.m;
    let orig_m = orig_cm.m;
    // orig_clen: MATP_MP contributes 2, MATL_ML/MATR_MR contribute 1 (cm_submodel.c:94-101).
    let mut orig_clen = 0i32;
    for v in 0..=(orig_m as usize) {
        let s = stid_at(orig_cm, v);
        if s == MATP_MP {
            orig_clen += 2;
        } else if s == MATL_ML || s == MATR_MR {
            orig_clen += 1;
        }
    }
    CMSubMap {
        spos: sstruct,
        epos: estruct,
        sstruct,
        estruct,
        s2o_smap: vec![[-1, -1]; (sub_m + 1) as usize],
        o2s_smap: vec![[-1, -1]; (orig_m + 1) as usize],
        s2o_id: vec![false; (sub_m + 1) as usize],
        sub_clen: estruct - sstruct + 1,
        orig_clen,
        sub_m,
        orig_m,
    }
}

/// C: map_orig2sub_cm_helper (cm_submodel.c:419). Register the (orig_v, sub_v)
/// mapping in both directions, skipping detached inserts and MATP type-mismatch,
/// and no-ops if already present. Returns true if a new mapping was made.
fn map_orig2sub_cm_helper(
    orig_cm: &CM,
    sub_cm: &CM,
    submap: &mut CMSubMap,
    orig_v: i32,
    sub_v: i32,
) -> bool {
    if orig_v == -1 || sub_v == -1 {
        return false;
    }
    let ov = orig_v as usize;
    let sv = sub_v as usize;
    // already have this mapping?
    if submap.o2s_smap[ov][0] == sub_v || submap.o2s_smap[ov][1] == sub_v {
        return false;
    }
    let orig_nd = orig_cm.ndidx[ov] as usize;
    let sub_nd = sub_cm.ndidx[sv] as usize;

    if sub_cm.sttype[sv] as i32 == IL_ST || sub_cm.sttype[sv] as i32 == IR_ST {
        // Skip if either is a detached insert (v+1 is an E state).
        if sttype_at(orig_cm, ov + 1) == E_ST || sttype_at(sub_cm, sv + 1) == E_ST {
            return false;
        }
    }
    // A sub_cm MATP_nd must map to an orig MATP_nd; refuse cross-type within MATP.
    if sub_cm.ndtype[sub_nd] as i32 == MATP_ND
        && orig_cm.ndtype[orig_nd] as i32 == MATP_ND
        && sub_cm.sttype[sv] != orig_cm.sttype[ov]
    {
        return false;
    }

    // Fill o2s_smap.
    if submap.o2s_smap[ov][0] == -1 {
        if submap.o2s_smap[ov][1] != -1 {
            panic!("map_orig2sub_cm_helper: o2s_smap[{}][0]==-1 but [1]!=-1", ov);
        }
        submap.o2s_smap[ov][0] = sub_v;
    } else if submap.o2s_smap[ov][1] != -1 {
        if submap.o2s_smap[ov][0] == sub_v || submap.o2s_smap[ov][1] == sub_v {
            return false;
        }
        panic!("map_orig2sub_cm_helper: o2s_smap[{}][0] and [1] both set", ov);
    } else {
        if submap.o2s_smap[ov][0] == sub_v || submap.o2s_smap[ov][1] == sub_v {
            return false;
        }
        submap.o2s_smap[ov][1] = sub_v;
    }

    // Fill s2o_smap.
    if submap.s2o_smap[sv][0] == -1 {
        if submap.s2o_smap[sv][1] != -1 {
            panic!("map_orig2sub_cm_helper: s2o_smap[{}][0]==-1 but [1]!=-1", sv);
        }
        submap.s2o_smap[sv][0] = orig_v;
    } else if submap.s2o_smap[sv][1] != -1 {
        panic!("map_orig2sub_cm_helper: s2o_smap[{}][0] and [1] both set", sv);
    } else {
        submap.s2o_smap[sv][1] = orig_v;
    }
    true
}

/// C: cm2sub_cm_check_id_next_node (cm_submodel.c:2525). If the orig/sub nodes
/// for this column AND the next column are the same type and consensus-aligned,
/// mark all (non-detached-insert) states of the sub node as identity-mapped
/// (s2o_id=TRUE), a time-saver for the later parameter copy.
#[allow(clippy::too_many_arguments)]
fn cm2sub_cm_check_id_next_node(
    orig_cm: &CM,
    sub_cm: &CM,
    orig_nd: i32,
    sub_nd: i32,
    submap: &mut CMSubMap,
    orig_cp9map: &CP9Map,
    sub_cp9map: &CP9Map,
) -> bool {
    if (orig_nd + 1) > (orig_cm.nodes - 1) {
        return false;
    }
    if (sub_nd + 1) > (sub_cm.nodes - 1) {
        return false;
    }
    if orig_cm.ndtype[orig_nd as usize] != sub_cm.ndtype[sub_nd as usize] {
        return false;
    }
    if orig_cm.ndtype[(orig_nd + 1) as usize] != sub_cm.ndtype[(sub_nd + 1) as usize] {
        return false;
    }
    let mut left_check = false;
    let mut right_check = false;
    let ol = orig_cp9map.nd2lpos[(orig_nd + 1) as usize];
    let sl = sub_cp9map.nd2lpos[(sub_nd + 1) as usize];
    let or = orig_cp9map.nd2rpos[(orig_nd + 1) as usize];
    let sr = sub_cp9map.nd2rpos[(sub_nd + 1) as usize];
    if ol == -1 && sl == -1 {
        left_check = true;
    }
    if ol == (sl + submap.spos - 1) {
        left_check = true;
    }
    if or == -1 && sr == -1 {
        right_check = true;
    }
    if or == (sr + submap.spos - 1) {
        right_check = true;
    }
    if left_check && right_check {
        let mut v_s = sub_cm.nodemap[sub_nd as usize];
        while sub_cm.ndidx[v_s as usize] == sub_nd {
            // if v+1 is an E_st, v is a detached insert: don't set s2o_id.
            if sttype_at(sub_cm, (v_s + 1) as usize) != E_ST {
                submap.s2o_id[v_s as usize] = true;
            }
            v_s += 1;
            if v_s >= sub_cm.m {
                break;
            }
        }
        return true;
    }
    false
}

/// C: map_orig2sub_cm (cm_submodel.c:228). Determine the orig<->sub state maps
/// (via each CM's CP9 map). B/E/S states (except ROOT_S) are intentionally left
/// unmapped; their transitions are handled specially later.
pub fn map_orig2sub_cm(orig_cm: &CM, sub_cm: &CM, submap: &mut CMSubMap) {
    let orig_cp9map = cp9_map_cm2hmm(orig_cm);
    let sub_cp9map = cp9_map_cm2hmm(sub_cm);
    let spos = submap.spos;
    let epos = submap.epos;

    // ROOT_S <-> ROOT_S (cm_submodel.c:285-287).
    map_orig2sub_cm_helper(orig_cm, sub_cm, submap, 0, 0);

    // ROOT_IL inserts before orig cc spos (cm_submodel.c:289-294). k_s=1 (insert).
    for x in 0..=1usize {
        for y in 0..=1usize {
            let ov = orig_cp9map.hns2cs[(spos - 1) as usize][1][x];
            let sv = sub_cp9map.hns2cs[0][1][y];
            map_orig2sub_cm_helper(orig_cm, sub_cm, submap, ov, sv);
        }
    }
    // ROOT_IR inserts after orig cc epos (cm_submodel.c:296-301).
    for x in 0..=1usize {
        for y in 0..=1usize {
            let ov = orig_cp9map.hns2cs[epos as usize][1][x];
            let sv = sub_cp9map.hns2cs[(epos - spos + 1) as usize][1][y];
            map_orig2sub_cm_helper(orig_cm, sub_cm, submap, ov, sv);
        }
    }

    for sub_k in 1..=sub_cp9map.hmm_m {
        let orig_k = sub_k + spos - 1;
        let sub_nd = sub_cp9map.pos2nd[sub_k as usize];
        let orig_nd = orig_cp9map.pos2nd[orig_k as usize];
        cm2sub_cm_check_id_next_node(
            orig_cm, sub_cm, orig_nd, sub_nd, submap, &orig_cp9map, &sub_cp9map,
        );
        for k_s in 0..3usize {
            for x in 0..=1usize {
                for y in 0..=1usize {
                    let ov = orig_cp9map.hns2cs[orig_k as usize][k_s][x];
                    let sv = sub_cp9map.hns2cs[sub_k as usize][k_s][y];
                    map_orig2sub_cm_helper(orig_cm, sub_cm, submap, ov, sv);
                }
            }
        }
    }
}

// =============================================================================
// Stage 3c: sub-CM probability construction (the FP-order-sensitive core).
// C cm_submodel.c: cm2sub_cm_emit_probs (1031), cm2sub_cm_trans_probs (1132) +
// _S (1220) + _B_E (1487), cm2sub_cm_add_single_trans (1644),
// cm2sub_cm_sum_subpaths (1715), cm2sub_cm_subtract_root_subpaths (2305),
// cm_trans_check (2220), orchestrated by build_sub_cm (563). find_impossible_*
// are NOT on this path (only called from the validation check_sub_cm).
// Symbolic constants only. FP: sub_psi/orig_psi are f64; sub_cm.t/.e are f32;
// C's exact float-vs-double promotions and accumulation order are reproduced.
// =============================================================================

/// C: esl_vec_FSum (esl_vectorops.c) — Kahan summation in f32.
fn esl_vec_fsum(v: &[f32]) -> f32 {
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

/// C: esl_vec_FNorm — normalize in place by the Kahan sum (per-element f32
/// division), or set uniform 1/n if the sum is 0.
fn esl_vec_fnorm(v: &mut [f32]) {
    let sum = esl_vec_fsum(v);
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

/// C: cm2sub_cm_emit_probs (cm_submodel.c:1031). Fill sub_cm emission probs for
/// state v_s from the 1 or 2 orig_cm states (v_o1, v_o2) it maps to, weighting
/// by orig_psi when two states (MP + ML/MR) collapse to a singlet.
fn cm2sub_cm_emit_probs(
    orig_cm: &CM,
    sub_cm: &mut CM,
    orig_psi: &[f64],
    v_s: usize,
    v_o1: i32,
    v_o2: i32,
    submap: &CMSubMap,
) {
    let k = ALPHABET_SIZE;
    if v_o1 == -1 {
        panic!("cm2sub_cm_emit_probs: sub state {} maps to 0 orig states", v_s);
    }
    let vo1 = v_o1 as usize;

    if sub_cm.sttype[v_s] as i32 == MP_ST {
        for i in 0..(k * k) {
            sub_cm.e[v_s][i] = orig_cm.e[vo1][i];
        }
        return;
    }
    if submap.s2o_id[v_s] {
        // singlet emitter, identity copy
        for i in 0..k {
            sub_cm.e[v_s][i] = orig_cm.e[vo1][i];
        }
        return;
    }
    // v_s is a singlet emitter. Two orig states map here only when one is MP.
    if orig_cm.sttype[vo1] as i32 == MP_ST {
        let is_left = if orig_cm.sttype[v_o2 as usize] as i32 == ML_ST {
            true
        } else if orig_cm.sttype[v_o2 as usize] as i32 == MR_ST {
            false
        } else {
            panic!("cm2sub_cm_emit_probs: v_s {} maps to MP + non-ML/MR", v_s);
        };
        for i in 0..k {
            if is_left {
                // C: for j=i*K..(i+1)*K: e[i] += psi[vo1]*orig_e[vo1][j]
                for j in (i * k)..((i + 1) * k) {
                    let acc = sub_cm.e[v_s][i] as f64 + orig_psi[vo1] * orig_cm.e[vo1][j] as f64;
                    sub_cm.e[v_s][i] = acc as f32;
                }
            } else {
                // C: for j=i; j<K*K; j+=K
                let mut j = i;
                while j < k * k {
                    let acc = sub_cm.e[v_s][i] as f64 + orig_psi[vo1] * orig_cm.e[vo1][j] as f64;
                    sub_cm.e[v_s][i] = acc as f32;
                    j += k;
                }
            }
        }
        if orig_cm.sttype[v_o2 as usize] as i32 == MP_ST {
            panic!("cm2sub_cm_emit_probs: v_s {} maps to two MATP_MP", v_s);
        }
        let vo2 = v_o2 as usize;
        for i in 0..k {
            let acc = sub_cm.e[v_s][i] as f64 + orig_psi[vo2] * orig_cm.e[vo2][i] as f64;
            sub_cm.e[v_s][i] = acc as f32;
        }
        esl_vec_fnorm(&mut sub_cm.e[v_s][0..k]);
        return;
    } else if v_o2 != -1 {
        panic!("cm2sub_cm_emit_probs: v_s {} maps to two states, neither MP", v_s);
    }
    // single singlet emitter
    for i in 0..k {
        sub_cm.e[v_s][i] = orig_cm.e[vo1][i];
    }
}

/// C: cm_trans_check (cm_submodel.c:2220). TRUE iff a transition exists from
/// state a to state b (b is within a's children).
fn cm_trans_check(cm: &CM, a: i32, b: i32) -> bool {
    if (a == -1 || b == -1) || (a > b) {
        return false;
    }
    (b - cm.cfirst[a as usize]) < cm.cnum[a as usize]
}

/// C: cm2sub_cm_sum_subpaths (cm_submodel.c:1715). Summed probability of all
/// subpaths in orig_cm from min(orig_v,orig_y) to max, ignoring subpaths that
/// pass through orig_cm inserts mapping to sub_cm inserts in init_sub_start's
/// node. Recursive. Returns f32; internal accumulation is f64 (C `sub_psi`).
#[allow(clippy::too_many_arguments)]
fn cm2sub_cm_sum_subpaths(
    orig_cm: &CM,
    sub_cm: &CM,
    submap: &CMSubMap,
    orig_v: i32,
    orig_y: i32,
    init_sub_start: i32,
    tmap: &Tmap,
    orig_psi: &[f64],
) -> f32 {
    if orig_v == -1 || orig_y == -1 {
        return 0.0;
    }
    let mut start = orig_v;
    let mut end = orig_y;
    if start > end {
        start = orig_y;
        end = orig_v;
    }
    if start == end {
        return orig_cm.t[start as usize][0]; // self-insert probability
    }
    let n = (end - start + 1) as usize;
    let mut sub_psi = vec![0.0f64; n];
    sub_psi[0] = 1.0; // must start in "start"

    let sstart = start as usize;
    let ssend = end as usize;
    let sub_start1 = submap.o2s_smap[sstart][0];
    let sub_start2 = submap.o2s_smap[sstart][1];
    let sub_end1 = submap.o2s_smap[ssend][0];
    let sub_end2 = submap.o2s_smap[ssend][1];

    // sub_cm inserts in init_sub_start's node.
    let init_sub_nd = sub_cm.ndidx[init_sub_start as usize] as usize;
    let mut sub_insert1 = -1i32;
    let mut sub_insert2 = -1i32;
    let ndt = sub_cm.ndtype[init_sub_nd] as i32;
    if ndt == MATP_ND {
        sub_insert1 = sub_cm.nodemap[init_sub_nd] + 4; // MATP_IL
        sub_insert2 = sub_cm.nodemap[init_sub_nd] + 5; // MATP_IR
    } else if ndt == ROOT_ND {
        sub_insert1 = 1; // ROOT_IL
        sub_insert2 = 2; // ROOT_IR
    } else if ndt == MATL_ND || ndt == MATR_ND || ndt == BEGR_ND {
        sub_insert1 = sub_cm.cfirst[init_sub_start as usize]; // MAT{L,R}_I* or BEGR_IL
    }
    let orig_insert1 = if sub_insert1 != -1 {
        submap.s2o_smap[sub_insert1 as usize][0]
    } else {
        -1
    };
    let orig_insert2 = if sub_insert2 != -1 {
        submap.s2o_smap[sub_insert2 as usize][0]
    } else {
        -1
    };

    for v in (start + 1)..=end {
        let vi = (v - start) as usize;
        sub_psi[vi] = 0.0;
        let vu = v as usize;
        let is_insert = orig_cm.sttype[vu] as i32 == IL_ST || orig_cm.sttype[vu] as i32 == IR_ST;
        if orig_cm.sttype[vu] as i32 == S_ST {
            // prev is BIF_B or END_E, no transition into S: carry forward
            sub_psi[vi] = sub_psi[vi - 1];
        }
        let mut skip_flag = false;
        if v != end && v == orig_insert1 && sub_insert1 >= init_sub_start {
            if cm_trans_check(sub_cm, sub_insert1, sub_end1)
                || cm_trans_check(sub_cm, sub_insert1, sub_end2)
                || cm_trans_check(sub_cm, sub_insert1, sub_start1)
                || cm_trans_check(sub_cm, sub_insert1, sub_start2)
            {
                skip_flag = true;
            }
        } else if v != end && v == orig_insert2 && sub_insert2 >= init_sub_start {
            if cm_trans_check(sub_cm, sub_insert2, sub_end1)
                || cm_trans_check(sub_cm, sub_insert2, sub_end2)
                || cm_trans_check(sub_cm, sub_insert2, sub_start1)
                || cm_trans_check(sub_cm, sub_insert2, sub_start2)
            {
                skip_flag = true;
            }
        }
        if !skip_flag {
            let isins = if is_insert { 1i32 } else { 0i32 };
            let mut y = orig_cm.pnum[vu] - 1;
            while y >= isins {
                let x = orig_cm.plast[vu] - y; // parent of v
                let tmap_val = tmap[orig_cm.stid[x as usize] as usize]
                    [orig_cm.ndtype[(orig_cm.ndidx[vu] + isins) as usize] as usize]
                    [orig_cm.stid[vu] as usize];
                if (x - start) < 0 {
                    // sub_psi[vi] += 0.
                } else {
                    sub_psi[vi] += sub_psi[(x - start) as usize]
                        * orig_cm.t[x as usize][tmap_val as usize] as f64;
                }
                y -= 1;
            }
            if v != end && is_insert {
                // self loop contribution; the factor is computed in f32 then promoted.
                let t0 = orig_cm.t[vu][0];
                let factor = t0 / (1.0f32 - t0);
                sub_psi[vi] += sub_psi[vi] * factor as f64;
            }
        }
    }
    let mut to_return = sub_psi[(end - start) as usize] as f32;

    // Ignore prob mass into 'start' / out of 'end' via inserts outside [start..end].
    let mut insert_to_start = 0.0f32;
    let mut end_to_insert = 0.0f32;
    let start_is_ins =
        orig_cm.sttype[sstart] as i32 == IL_ST || orig_cm.sttype[sstart] as i32 == IR_ST;
    let end_is_ins = orig_cm.sttype[ssend] as i32 == IL_ST || orig_cm.sttype[ssend] as i32 == IR_ST;
    if !start_is_ins && !end_is_ins {
        if orig_insert1 != -1 && orig_insert1 < start {
            // start is not insert here => self_loop_factor = 1.0
            let self_loop_factor = 1.0f64;
            let sp = cm2sub_cm_sum_subpaths(
                orig_cm, sub_cm, submap, orig_insert1, start, init_sub_start, tmap, orig_psi,
            );
            insert_to_start =
                (insert_to_start as f64 + self_loop_factor * orig_psi[orig_insert1 as usize] * sp as f64) as f32;
        }
        if orig_insert1 != -1 && orig_insert1 > end {
            let sp = cm2sub_cm_sum_subpaths(
                orig_cm, sub_cm, submap, end, orig_insert1, init_sub_start, tmap, orig_psi,
            );
            end_to_insert += sp;
        }
        if orig_insert2 != -1 && orig_insert2 < start {
            let self_loop_factor = 1.0f64;
            let sp = cm2sub_cm_sum_subpaths(
                orig_cm, sub_cm, submap, orig_insert2, start, init_sub_start, tmap, orig_psi,
            );
            insert_to_start =
                (insert_to_start as f64 + self_loop_factor * orig_psi[orig_insert2 as usize] * sp as f64) as f32;
        }
        if orig_insert2 != -1 && orig_insert2 > end {
            let sp = cm2sub_cm_sum_subpaths(
                orig_cm, sub_cm, submap, end, orig_insert2, init_sub_start, tmap, orig_psi,
            );
            end_to_insert += sp;
        }
    }
    // to_return *= (1 - insert_to_start/orig_psi[start]); to_return *= (1 - end_to_insert)
    to_return = (to_return as f64 * (1.0 - (insert_to_start as f64 / orig_psi[sstart]))) as f32;
    to_return = (to_return as f64 * (1.0 - end_to_insert as f64)) as f32;
    to_return
}

/// C: cm2sub_cm_add_single_trans (cm_submodel.c:1644). Add a virtual-count
/// contribution to a single sub_cm transition, from orig_v/orig_y via subpaths.
#[allow(clippy::too_many_arguments)]
fn cm2sub_cm_add_single_trans(
    orig_cm: &CM,
    sub_cm: &mut CM,
    submap: &CMSubMap,
    orig_v: i32,
    orig_y: i32,
    sub_v: usize,
    yoffset: usize,
    orig_psi: &[f64],
    tmap: &Tmap,
) {
    if orig_v == -1 || orig_y == -1 {
        return;
    }
    let start = if orig_y < orig_v { orig_y } else { orig_v };
    let sp = cm2sub_cm_sum_subpaths(
        orig_cm, sub_cm, submap, orig_v, orig_y, sub_v as i32, tmap, orig_psi,
    );
    let acc = sub_cm.t[sub_v][yoffset] as f64 + orig_psi[start as usize] * sp as f64;
    sub_cm.t[sub_v][yoffset] = acc as f32;
}

/// C: cm2sub_cm_trans_probs (cm_submodel.c:1132). Fill virtual counts for the
/// transitions out of sub_cm state v_s (non-B/S/E, handled elsewhere).
fn cm2sub_cm_trans_probs(
    orig_cm: &CM,
    sub_cm: &mut CM,
    orig_psi: &[f64],
    tmap: &Tmap,
    v_s: usize,
    submap: &CMSubMap,
) {
    if submap.s2o_id[v_s] {
        let v_o = submap.s2o_smap[v_s][0] as usize;
        let cnum = sub_cm.cnum[v_s] as usize;
        for yoffset in 0..cnum {
            sub_cm.t[v_s][yoffset] = (orig_psi[v_o] * orig_cm.t[v_o][yoffset] as f64) as f32;
        }
        return;
    }
    let cnum = sub_cm.cnum[v_s] as usize;
    let cfirst = sub_cm.cfirst[v_s];
    let stt = sub_cm.sttype[v_s] as i32;

    let v_o = submap.s2o_smap[v_s][0];
    if v_o == -1 {
        if stt != S_ST && stt != E_ST && stt != B_ST {
            panic!("cm2sub_cm_trans_probs: v_s {} maps to no state but isn't B/E/S", v_s);
        }
    } else {
        if (stt == S_ST || stt == E_ST || stt == B_ST) && v_s != 0 {
            panic!("cm2sub_cm_trans_probs: v_s {} is S/E/B but maps to orig {}", v_s, v_o);
        }
        for yoffset in 0..cnum {
            let y_s = (cfirst + yoffset as i32) as usize;
            if sttype_at(sub_cm, y_s + 1) != E_ST {
                let (a0, a1) = (submap.s2o_smap[y_s][0], submap.s2o_smap[y_s][1]);
                cm2sub_cm_add_single_trans(orig_cm, sub_cm, submap, v_o, a0, v_s, yoffset, orig_psi, tmap);
                cm2sub_cm_add_single_trans(orig_cm, sub_cm, submap, v_o, a1, v_s, yoffset, orig_psi, tmap);
            }
        }
    }
    let v_o = submap.s2o_smap[v_s][1];
    if v_o != -1 {
        for yoffset in 0..cnum {
            let y_s = (cfirst + yoffset as i32) as usize;
            if sttype_at(sub_cm, y_s + 1) != E_ST {
                let (a0, a1) = (submap.s2o_smap[y_s][0], submap.s2o_smap[y_s][1]);
                cm2sub_cm_add_single_trans(orig_cm, sub_cm, submap, v_o, a0, v_s, yoffset, orig_psi, tmap);
                cm2sub_cm_add_single_trans(orig_cm, sub_cm, submap, v_o, a1, v_s, yoffset, orig_psi, tmap);
            }
        }
    }
}

/// C: cm2sub_cm_subtract_root_subpaths (cm_submodel.c:2305). Correct ROOT_S and
/// ROOT_IL virtual counts for subpaths double-counted via the two ROOT inserts.
fn cm2sub_cm_subtract_root_subpaths(
    orig_cm: &CM,
    sub_cm: &mut CM,
    orig_psi: &[f64],
    tmap: &Tmap,
    submap: &CMSubMap,
) {
    let sub_il = 1usize; // ROOT_IL
    let sub_ir = 2usize; // ROOT_IR
    let orig_il = submap.s2o_smap[sub_il][0];
    let orig_ir = submap.s2o_smap[sub_ir][0];
    let orig_ss = submap.s2o_smap[3][0];

    // split-set guarantee check (cm_submodel.c:2337-2347).
    let cnum0 = sub_cm.cnum[0] as usize;
    let cfirst0 = sub_cm.cfirst[0];
    for yoffset in 2..cnum0 {
        let sub_y = (yoffset as i32 + cfirst0) as usize;
        let orig_y = submap.s2o_smap[sub_y][0];
        if (orig_il > orig_ss && orig_il < orig_y)
            || (orig_il < orig_ss && orig_il > orig_y)
            || (orig_ir > orig_ss && orig_ir < orig_y)
            || (orig_ir < orig_ss && orig_ir > orig_y)
        {
            panic!("cm2sub_cm_subtract_root_subpaths: split-set guarantee violated");
        }
    }

    // Adjust ROOT_S -> ROOT_IR (t[0][1]) for ir->il (cases 2A/2B/2C).
    if orig_ir < orig_il {
        let sp = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir, orig_il, 0, tmap, orig_psi);
        let acc = sub_cm.t[0][1] as f64 - orig_psi[orig_ir as usize] * sp as f64;
        sub_cm.t[0][1] = acc as f32;
    }

    for yoffset in 0..cnum0 {
        let sub_y = (cfirst0 + yoffset as i32) as usize;
        let orig_ss1 = submap.s2o_smap[sub_y][0];
        let orig_ss2 = submap.s2o_smap[sub_y][1];
        if sub_cm.ndidx[sub_y] != 0 {
            // case 2B: ir < ss < il
            if orig_ir < orig_ss1 && orig_ss1 < orig_il {
                let a = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir, orig_ss1, sub_il as i32, tmap, orig_psi);
                let b = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ss1, orig_il, sub_il as i32, tmap, orig_psi);
                let acc = sub_cm.t[sub_il][yoffset] as f64 - orig_psi[orig_ir as usize] * a as f64 * b as f64;
                sub_cm.t[sub_il][yoffset] = acc as f32;
                if orig_ss2 != -1 {
                    let a2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir, orig_ss2, sub_il as i32, tmap, orig_psi);
                    let b2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ss2, orig_il, sub_il as i32, tmap, orig_psi);
                    let acc = sub_cm.t[1][yoffset] as f64 - orig_psi[orig_ir as usize] * a2 as f64 * b2 as f64;
                    sub_cm.t[1][yoffset] = acc as f32;
                }
            }
            // case 1C: ss < il < ir
            if orig_ss1 < orig_il && orig_il < orig_ir {
                let a = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ss1, orig_il, sub_il as i32, tmap, orig_psi);
                let t0 = orig_cm.t[orig_il as usize][0];
                let factor = 1.0f64 + (t0 / (1.0f32 - t0)) as f64;
                let b = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_il, orig_ir, sub_il as i32, tmap, orig_psi);
                let acc = sub_cm.t[sub_il][yoffset] as f64
                    - orig_psi[orig_ss1 as usize] * a as f64 * factor * b as f64;
                sub_cm.t[sub_il][yoffset] = acc as f32;
                if orig_ss2 != -1 {
                    let a2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ss2, orig_il, sub_il as i32, tmap, orig_psi);
                    let b2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_il, orig_ir, sub_il as i32, tmap, orig_psi);
                    let acc = sub_cm.t[sub_il][yoffset] as f64
                        - orig_psi[orig_ss2 as usize] * a2 as f64 * factor * b2 as f64;
                    sub_cm.t[sub_il][yoffset] = acc as f32;
                }
            }
            // case 2A: ir < il < ss
            if orig_ir < orig_il && orig_il < orig_ss1 {
                let a = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir, orig_il, sub_il as i32, tmap, orig_psi);
                let t0 = orig_cm.t[orig_il as usize][0];
                let factor = 1.0f64 + (t0 / (1.0f32 - t0)) as f64;
                let b = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_il, orig_ss1, sub_il as i32, tmap, orig_psi);
                let acc = sub_cm.t[sub_il][yoffset] as f64
                    - orig_psi[orig_ir as usize] * a as f64 * factor * b as f64;
                sub_cm.t[sub_il][yoffset] = acc as f32;
                if orig_ss2 != -1 {
                    let a2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir, orig_il, sub_il as i32, tmap, orig_psi);
                    let b2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_il, orig_ss2, sub_il as i32, tmap, orig_psi);
                    let acc = sub_cm.t[sub_il][yoffset] as f64
                        - orig_psi[orig_ir as usize] * a2 as f64 * factor * b2 as f64;
                    sub_cm.t[sub_il][yoffset] = acc as f32;
                }
            }
            // case 1B: il < ss < ir
            if orig_il < orig_ss1 && orig_ss1 < orig_ir {
                let a = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_il, orig_ss1, sub_il as i32, tmap, orig_psi);
                let b = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ss1, orig_ir, sub_il as i32, tmap, orig_psi);
                let acc = sub_cm.t[sub_il][yoffset] as f64 - orig_psi[orig_il as usize] * a as f64 * b as f64;
                sub_cm.t[sub_il][yoffset] = acc as f32;
                if orig_ss2 != -1 {
                    let a2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_il, orig_ss2, sub_il as i32, tmap, orig_psi);
                    let b2 = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ss2, orig_ir, sub_il as i32, tmap, orig_psi);
                    let acc = sub_cm.t[sub_il][yoffset] as f64 - orig_psi[orig_il as usize] * a2 as f64 * b2 as f64;
                    sub_cm.t[sub_il][yoffset] = acc as f32;
                }
            }
        }
    }
}

/// C: cm2sub_cm_trans_probs_S (cm_submodel.c:1220). Fill virtual counts for
/// transitions out of a sub_cm S state (BEGL/BEGR/ROOT).
fn cm2sub_cm_trans_probs_S(
    orig_cm: &CM,
    sub_cm: &mut CM,
    orig_psi: &[f64],
    tmap: &Tmap,
    v_start: usize,
    submap: &CMSubMap,
) {
    let sub_nd = sub_cm.ndidx[v_start] as usize;
    let ndt = sub_cm.ndtype[sub_nd] as i32;
    let next_is_bif = sub_cm.ndtype[sub_nd + 1] as i32 == BIF_ND;

    if ndt == BEGL_ND {
        if next_is_bif {
            sub_cm.t[v_start][0] = 1.0; // BEGL_S -> BIF_B
        } else {
            let cnum = sub_cm.cnum[v_start] as usize;
            let cfirst = sub_cm.cfirst[v_start];
            for yoffset in 0..cnum {
                let y_s = (cfirst + yoffset as i32) as usize;
                sub_cm.t[v_start][yoffset] = orig_psi[submap.s2o_smap[y_s][0] as usize] as f32;
                if submap.s2o_smap[y_s][1] != -1 {
                    let acc = sub_cm.t[v_start][yoffset] as f64
                        + orig_psi[submap.s2o_smap[y_s][1] as usize];
                    sub_cm.t[v_start][yoffset] = acc as f32;
                }
            }
        }
    } else if ndt == BEGR_ND {
        let v_s_insert = v_start + 1;
        let cnum_ins = sub_cm.cnum[v_s_insert] as usize;
        if next_is_bif {
            let v_o_insert = submap.s2o_smap[v_s_insert][0] as usize;
            if submap.s2o_smap[v_s_insert][1] != -1 {
                panic!("BEGR_IL maps to 2 orig states (unimplemented)");
            }
            // BEGR_IL -> BIF_B (t[1]) from orig self-loop complement.
            sub_cm.t[v_s_insert][1] =
                (orig_psi[v_o_insert] * (1.0 - orig_cm.t[v_o_insert][0] as f64)) as f32;
            esl_vec_fnorm(&mut sub_cm.t[v_s_insert][0..cnum_ins]);
            let il_psi = orig_psi[submap.s2o_smap[v_s_insert][0] as usize] as f32;
            sub_cm.t[v_start][0] = ((1.0 - sub_cm.t[v_s_insert][0] as f64) * il_psi as f64) as f32;
            sub_cm.t[v_start][1] = (1.0 - sub_cm.t[v_start][0] as f64) as f32;
        } else {
            esl_vec_fnorm(&mut sub_cm.t[v_s_insert][0..cnum_ins]);
            let il_psi = orig_psi[submap.s2o_smap[v_s_insert][0] as usize] as f32;
            sub_cm.t[v_start][0] = ((1.0 - sub_cm.t[v_s_insert][0] as f64) * il_psi as f64) as f32;
            let mut sum = sub_cm.t[v_start][0];
            let cnum = sub_cm.cnum[v_start] as usize;
            let cfirst = sub_cm.cfirst[v_start];
            for yoffset in 1..cnum {
                let y_s = (cfirst + yoffset as i32) as usize;
                let mut temp_psi = orig_psi[submap.s2o_smap[y_s][0] as usize] as f32;
                if submap.s2o_smap[y_s][1] != -1 {
                    temp_psi = (temp_psi as f64 + orig_psi[submap.s2o_smap[y_s][1] as usize]) as f32;
                }
                // temp_psi - il_psi * t[v_s_insert][yoffset]   (all f32)
                sub_cm.t[v_start][yoffset] = temp_psi - il_psi * sub_cm.t[v_s_insert][yoffset];
                sum += sub_cm.t[v_start][yoffset];
            }
            if (sum < 1.0 && (1.0 - sum) > 0.001) || (sum > 1.0 && (sum - 1.0) > 0.001) {
                panic!("cm2sub_cm_trans_probs_S: BEGR_S transitions sum {} off", sum);
            }
        }
    } else if ndt == ROOT_ND && next_is_bif {
        let orig_il1 = submap.s2o_smap[1][0];
        let orig_il2 = submap.s2o_smap[1][1];
        let orig_ir1 = submap.s2o_smap[2][0];
        let orig_ir2 = submap.s2o_smap[2][1];
        if orig_ir1 < orig_il1 {
            let sp = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir1, orig_il1, 0, tmap, orig_psi);
            sub_cm.t[0][1] = (sub_cm.t[0][1] as f64 - orig_psi[orig_ir1 as usize] * sp as f64) as f32;
        }
        if orig_ir2 != -1 && orig_ir2 < orig_il1 {
            let sp = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir2, orig_il1, 0, tmap, orig_psi);
            sub_cm.t[0][1] = (sub_cm.t[0][1] as f64 - orig_psi[orig_ir2 as usize] * sp as f64) as f32;
        }
        if orig_il2 != -1 && orig_ir1 < orig_il2 {
            let sp = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir1, orig_il2, 0, tmap, orig_psi);
            sub_cm.t[0][1] = (sub_cm.t[0][1] as f64 - orig_psi[orig_ir1 as usize] * sp as f64) as f32;
        }
        if orig_ir2 != -1 && orig_il2 != -1 && orig_ir2 < orig_il2 {
            let sp = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_ir2, orig_il2, 0, tmap, orig_psi);
            sub_cm.t[0][1] = (sub_cm.t[0][1] as f64 - orig_psi[orig_ir2 as usize] * sp as f64) as f32;
        }
        // ROOT_S -> BIF_B
        sub_cm.t[0][2] = (1.0 - (sub_cm.t[0][0] + sub_cm.t[0][1]) as f64) as f32;

        // ROOT_IL -> {IL, IR, BIF}
        let v_s_insert = v_start + 1;
        let mut v_o_insert = submap.s2o_smap[v_s_insert][0] as usize;
        let mut temp_psi_sum = orig_psi[v_o_insert] as f32;
        if submap.s2o_smap[v_s_insert][1] != -1 {
            v_o_insert = submap.s2o_smap[v_s_insert][1] as usize;
            temp_psi_sum = (temp_psi_sum as f64 + orig_psi[v_o_insert]) as f32;
        }
        sub_cm.t[v_s_insert][0] /= temp_psi_sum;
        sub_cm.t[v_s_insert][1] /= temp_psi_sum;
        sub_cm.t[v_s_insert][2] =
            (1.0 - (sub_cm.t[v_s_insert][0] + sub_cm.t[v_s_insert][1]) as f64) as f32;

        // ROOT_IR -> {IR, BIF}
        let v_s_insert = v_start + 2;
        let mut v_o_insert = submap.s2o_smap[v_s_insert][0] as usize;
        let mut temp_psi_sum = orig_psi[v_o_insert] as f32;
        if submap.s2o_smap[v_s_insert][1] != -1 {
            v_o_insert = submap.s2o_smap[v_s_insert][1] as usize;
            temp_psi_sum = (temp_psi_sum as f64 + orig_psi[v_o_insert]) as f32;
        }
        sub_cm.t[v_s_insert][0] /= temp_psi_sum;
        sub_cm.t[v_s_insert][1] = (1.0 - sub_cm.t[v_s_insert][0] as f64) as f32;
    }
    // ROOT_nd with non-BIF next: transitions already set, nothing to do.
}

/// C: cm2sub_cm_trans_probs_B_E (cm_submodel.c:1487). Fill virtual counts for
/// transitions INTO a sub_cm B or E state v_be from the previous node.
fn cm2sub_cm_trans_probs_B_E(
    orig_cm: &CM,
    sub_cm: &mut CM,
    orig_psi: &[f64],
    tmap: &Tmap,
    v_be: usize,
    submap: &CMSubMap,
) {
    let sub_nd = sub_cm.ndidx[v_be] as usize;
    let psub_nd = sub_nd - 1;
    let into_end_flag = sub_cm.sttype[v_be] as i32 == E_ST;
    let pndt = sub_cm.ndtype[psub_nd] as i32;

    if pndt == MATP_ND {
        let mut bif_end_yoffset = 2usize;
        let sub_il = (sub_cm.nodemap[psub_nd] + 4) as usize;
        let sub_ir = (sub_cm.nodemap[psub_nd] + 5) as usize;
        let orig_il = submap.s2o_smap[sub_il][0];
        let orig_ir = submap.s2o_smap[sub_ir][0];
        if into_end_flag && orig_ir != -1 {
            panic!("trans_probs_B_E: into_end but MATP_IR maps to orig");
        }
        if orig_ir == -1 && !into_end_flag {
            panic!("trans_probs_B_E: not into_end but MATP_IR unmapped");
        }
        for sub_v in (sub_cm.nodemap[psub_nd] as usize)..sub_il {
            let orig_v = submap.s2o_smap[sub_v][0] as usize;
            sub_cm.t[sub_v][bif_end_yoffset] =
                (orig_psi[orig_v] - (sub_cm.t[sub_v][0] + sub_cm.t[sub_v][1]) as f64) as f32;
        }
        sub_cm.t[sub_il][bif_end_yoffset] =
            (orig_psi[orig_il as usize] - (sub_cm.t[sub_il][0] + sub_cm.t[sub_il][1]) as f64) as f32;
        bif_end_yoffset = 1;
        if into_end_flag {
            sub_cm.t[sub_ir][bif_end_yoffset] = 1.0;
        } else {
            sub_cm.t[sub_ir][bif_end_yoffset] =
                (orig_psi[orig_ir as usize] - sub_cm.t[sub_ir][0] as f64) as f32;
        }
    } else if pndt == MATL_ND || pndt == MATR_ND {
        let bif_end_yoffset = 1usize;
        let sub_i = (sub_cm.nodemap[psub_nd] + 2) as usize;
        let orig_i = submap.s2o_smap[sub_i][0];
        if into_end_flag && orig_i != -1 {
            panic!("trans_probs_B_E: into_end but MAT*_I* maps to orig");
        }
        for sub_v in (sub_cm.nodemap[psub_nd] as usize)..sub_i {
            let orig_v1 = submap.s2o_smap[sub_v][0];
            let orig_v2 = submap.s2o_smap[sub_v][1];
            if into_end_flag {
                sub_cm.t[sub_v][bif_end_yoffset] = 1.0;
            } else {
                if orig_v1 < orig_i {
                    let contribution =
                        cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_v1, orig_i, sub_v as i32, tmap, orig_psi);
                    sub_cm.t[sub_v][bif_end_yoffset] =
                        (orig_psi[orig_v1 as usize] * (1.0 - contribution as f64)) as f32;
                } else {
                    let contribution: f32 = (orig_psi[orig_i as usize]
                        * cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_i, orig_v1, sub_v as i32, tmap, orig_psi) as f64)
                        as f32;
                    sub_cm.t[sub_v][bif_end_yoffset] = (orig_psi[orig_v1 as usize]
                        * (1.0 - (contribution as f64 / orig_psi[orig_v1 as usize])))
                        as f32;
                }
                if orig_v2 != -1 {
                    if orig_v2 < orig_i {
                        let contribution =
                            cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_v2, orig_i, sub_v as i32, tmap, orig_psi);
                        let acc = sub_cm.t[sub_v][bif_end_yoffset] as f64
                            + orig_psi[orig_v2 as usize] * (1.0 - contribution as f64);
                        sub_cm.t[sub_v][bif_end_yoffset] = acc as f32;
                    } else {
                        let contribution: f32 = (orig_psi[orig_i as usize]
                            * cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_i, orig_v2, sub_v as i32, tmap, orig_psi) as f64)
                            as f32;
                        let acc = sub_cm.t[sub_v][bif_end_yoffset] as f64
                            + orig_psi[orig_v2 as usize]
                                * (1.0 - (contribution as f64 / orig_psi[orig_v2 as usize]));
                        sub_cm.t[sub_v][bif_end_yoffset] = acc as f32;
                    }
                }
            }
        }
        if into_end_flag {
            sub_cm.t[sub_i][1] = 1.0;
        } else {
            let sp = cm2sub_cm_sum_subpaths(orig_cm, sub_cm, submap, orig_i, orig_i, sub_i as i32, tmap, orig_psi);
            sub_cm.t[sub_i][1] = (orig_psi[orig_i as usize] * (1.0 - sp as f64)) as f32;
        }
    }
    // ROOT/BEGL/BEGR previous nodes: handled in trans_probs_S.
}

/// C: build_sub_cm (cm_submodel.c:563). Build the parameterized sub-CM from the
/// (configured, global) original CM for consensus columns [spos..epos]. Returns
/// the sub-CM (count-normalized, not yet logoddsified) and its orig<->sub map.
pub fn build_sub_cm(orig_cm: &CM, spos: i32, epos: i32) -> (CM, CMSubMap) {
    // Note: C re-runs cm_find_and_detach_dual_inserts(orig_cm) here; on an
    // already-built (detached) CM it is a no-op, so orig_cm is untouched and
    // orig_psi is identical (verified byte-exact vs C previously).

    // Topology: CreateCMConsensus -> sub structure string -> ConsensusModelmaker
    // -> CMRebalance (cm_submodel.c:598-684).
    let mut sub_cm = build_sub_cm_topology(orig_cm, spos, epos);

    // Maps (cm_submodel.c:683).
    let mut submap = alloc_sub_map(&sub_cm, orig_cm, spos, epos);
    map_orig2sub_cm(orig_cm, &sub_cm, &mut submap);

    // orig_psi + transition map (cm_submodel.c:686-688).
    let orig_psi = cm_expected_state_occupancy(orig_cm);
    let tmap: Tmap = cm_create_transition_map();

    // CMZero + null + flags (cm_submodel.c:690-708). Only t/e-affecting bits.
    sub_cm.cm_zero();
    sub_cm.cm_set_null_model(&orig_cm.null);
    sub_cm.el_selfsc = orig_cm.el_selfsc;
    // C build_sub_cm: sub_cm->W = orig_cm->W if unset (cm_submodel.c:779).
    if sub_cm.w == 0 {
        sub_cm.w = orig_cm.w;
    }
    sub_cm.flags = crate::cm::CM_IS_SUB;

    let m = sub_cm.m as usize;

    // Emissions (cm_submodel.c:710-720).
    for v_s in 0..m {
        if sttype_at(&sub_cm, v_s + 1) == E_ST {
            // detached insert: equiprobable (irrelevant)
            esl_vec_fnorm(&mut sub_cm.e[v_s][0..ALPHABET_SIZE]);
        } else {
            let st = sub_cm.sttype[v_s] as i32;
            if st != S_ST && st != D_ST && st != B_ST && st != E_ST {
                let (o1, o2) = (submap.s2o_smap[v_s][0], submap.s2o_smap[v_s][1]);
                cm2sub_cm_emit_probs(orig_cm, &mut sub_cm, &orig_psi, v_s, o1, o2, &submap);
            }
        }
    }

    // Transitions for non-B/S/E states (cm_submodel.c:726-736).
    for v_s in 0..m {
        if sttype_at(&sub_cm, v_s + 1) == E_ST {
            let cnum = sub_cm.cnum[v_s] as usize;
            esl_vec_fnorm(&mut sub_cm.t[v_s][0..cnum]);
        } else {
            let st = sub_cm.sttype[v_s] as i32;
            if v_s == 0 || (st != S_ST && st != B_ST && st != E_ST) {
                cm2sub_cm_trans_probs(orig_cm, &mut sub_cm, &orig_psi, &tmap, v_s, &submap);
            }
        }
    }

    // Correct ROOT overcounting (cm_submodel.c:742-758).
    let nodes = sub_cm.nodes as usize;
    for n_s in 0..nodes {
        if sub_cm.ndtype[n_s] as i32 == MATP_ND
            && sttype_at(&sub_cm, (sub_cm.nodemap[n_s] + 5 + 1) as usize) != E_ST
        {
            let il1 = submap.s2o_smap[(sub_cm.nodemap[n_s] + 4) as usize][1];
            let ir1 = submap.s2o_smap[(sub_cm.nodemap[n_s] + 5) as usize][1];
            if il1 != -1 || ir1 != -1 {
                panic!("build_sub_cm: MATP_IL/IR node {} maps to 2 cm states", n_s);
            }
            let a = submap.s2o_smap[(sub_cm.nodemap[n_s] + 4) as usize][0];
            let b = submap.s2o_smap[(sub_cm.nodemap[n_s] + 5) as usize][0];
            if a != b - 1 {
                panic!("build_sub_cm: MATP_IL/IR node {} not adjacent orig states", n_s);
            }
        }
        if sub_cm.ndtype[n_s] as i32 == ROOT_ND && sub_cm.ndtype[n_s + 1] as i32 != BIF_ND {
            cm2sub_cm_subtract_root_subpaths(orig_cm, &mut sub_cm, &orig_psi, &tmap, &submap);
        }
    }

    // Transitions into E/B and out of S (cm_submodel.c:761-770).
    for v_s in 0..m {
        if sub_cm.sttype[v_s] as i32 == S_ST {
            cm2sub_cm_trans_probs_S(orig_cm, &mut sub_cm, &orig_psi, &tmap, v_s, &submap);
        }
        if sub_cm.sttype[v_s] as i32 == E_ST || sub_cm.sttype[v_s] as i32 == B_ST {
            cm2sub_cm_trans_probs_B_E(orig_cm, &mut sub_cm, &orig_psi, &tmap, v_s, &submap);
        }
    }

    // Detach sub_cm dual inserts, then renormalize (cm_submodel.c:776-798).
    cm_find_and_detach_dual_inserts(&mut sub_cm, false, true);
    sub_cm.cm_renormalize();

    (sub_cm, submap)
}

// =============================================================================
// Stage 4: sub-CM log-odds scoring. C SubCMLogoddsify (cm_submodel.c:3663) +
// SubFCalcAndCopyOptimizedEmitScoresFromMother (3838). For states that map
// identically to a mother state, C COPIES the mother's scores (bit-exact); for
// the rest it computes them exactly like standard CMLogoddsify. We reuse the
// byte-verified standard scoring for ALL states, then overwrite the identity
// states with the mother's scores — equivalent to C's copy-vs-compute split.
// oesc is a pure function of esc, so copying mother.oesc[mv] for id states is
// the same as recomputing from the copied esc (SubFCalc's optimization).
// =============================================================================

/// C: SubCMLogoddsify (cm_submodel.c:3663) + SubFCalc (3838). Fill the sub-CM's
/// log-odds scores from the (configured) mother CM and the orig<->sub map.
pub fn sub_cm_logoddsify(sub_cm: &mut CM, mother_cm: &CM, submap: &CMSubMap) {
    let k = ALPHABET_SIZE;
    // Standard (byte-verified) scoring for every state: tsc/esc/oesc/beginsc/
    // endsc + itsc/ioesc/ibeginsc/iendsc/iel_selfsc.
    crate::cm_nohmm::cm_configure_scores_global(sub_cm);

    // Overwrite identity-mapped states with the mother's scores (C copy path).
    for v in 0..sub_cm.m as usize {
        if !submap.s2o_id[v] {
            continue;
        }
        let mv = submap.s2o_smap[v][0] as usize;
        let st = sub_cm.sttype[v] as i32;
        if st != B_ST && st != E_ST {
            let cnum = sub_cm.cnum[v] as usize;
            for x in 0..cnum {
                sub_cm.tsc[v][x] = mother_cm.tsc[mv][x];
                sub_cm.itsc[v][x] = mother_cm.itsc[mv][x];
            }
        }
        if st == MP_ST {
            for i in 0..(k * k) {
                sub_cm.esc[v][i] = mother_cm.esc[mv][i];
            }
        } else if st == ML_ST || st == MR_ST || st == IL_ST || st == IR_ST {
            for i in 0..k {
                sub_cm.esc[v][i] = mother_cm.esc[mv][i];
            }
        }
        // Optimized emit scores: copy the whole vector (same state type => same
        // size). C SubFCalc copies mother_cm->oesc[mv]; ioesc from floats.
        if !mother_cm.oesc[mv].is_empty() {
            sub_cm.oesc[v] = mother_cm.oesc[mv].clone();
        }
        if !mother_cm.ioesc[mv].is_empty() {
            sub_cm.ioesc[v] = mother_cm.ioesc[mv].clone();
        }
        sub_cm.beginsc[v] = mother_cm.beginsc[mv];
        sub_cm.ibeginsc[v] = mother_cm.ibeginsc[mv];
        sub_cm.endsc[v] = mother_cm.endsc[mv];
        sub_cm.iendsc[v] = mother_cm.iendsc[mv];
    }
}

// =============================================================================
// Stage 5: sub-CM configuration + parsetree back-map.
// =============================================================================

/// C: InsertTraceNode (cm_parsetree.c:221) with TRMODE_J. Append a node to <tr>
/// as the left (or right) child of <parent> (parent=-1 initializes the root).
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

/// C: cm_ConfigureSub (cm_modelconfig.c:139), specialized to the only path
/// `cmalign --sub` uses: global (`-g`), non-truncated, HMM-banded. Fills the
/// sub-CM's log-odds scores (via [`sub_cm_logoddsify`]) and builds+configures its
/// CP9 HMM from the mother's CP9 for band calculation. Returns (sub cp9, sub
/// cp9map). W/QDB recomputation and the ml-p7 build are skipped: neither affects
/// the HMM-banded global alignment output (the el_selfsc*W clamp never trips for a
/// normal model, and cm_cp9_to_p7 is only used by the search filters).
pub fn configure_sub(
    sub_cm: &mut CM,
    orig_cm: &CM,
    orig_cp9: &CP9,
    submap: &CMSubMap,
) -> (CP9, CP9Map) {
    // Build the sub CP9 + map from the mother's CP9 (C cm_modelconfig.c:250).
    let (mut sub_cp9, sub_cp9map) = crate::cp9::sub_build_cp9_hmm_from_mother(
        sub_cm, orig_cm, orig_cp9, submap.spos, submap.epos,
    );
    // CM_CONFIG_SUB block (C cm_modelconfig.c:331-335): all start/end points
    // equiprobable, incl. 1 and M.
    let mm = sub_cp9.m as f32;
    let swfull = (mm - 1.0) / mm;
    crate::cp9::cp9_sw_config(&mut sub_cp9, swfull, swfull);
    // SubCMLogoddsify (C cm_modelconfig.c:349) — sub_cm bit scores.
    sub_cm_logoddsify(sub_cm, orig_cm, submap);
    // CP9Logoddsify(cm->cp9) (C cm_modelconfig.c:354).
    crate::cp9::cp9_logoddsify(&mut sub_cp9);
    (sub_cp9, sub_cp9map)
}

/// C: sub_cm2cm_parsetree (cm_submodel.c:3298). Convert a parsetree built against
/// <sub_cm> into a parsetree in the original CM's coordinate system, using the
/// s2o state maps. Faithful to the NON-local variant (local begins/ends off): any
/// orig node with no state visited in the sub parse defaults to its D (MAT*), S
/// (BEG*/ROOT) or E (END) state, and the emit map is recomputed from the
/// resulting per-node state assignments. Returns the orig-CM parsetree.
pub fn sub_cm2cm_parsetree(
    orig_cm: &CM,
    sub_cm: &CM,
    sub_tr: &Parsetree,
    submap: &CMSubMap,
) -> Parsetree {
    let nnodes = orig_cm.nodes as usize;

    // Per-orig-node bookkeeping (C 3354-3377).
    let mut ss_used = vec![-1i32; nnodes + 1];
    let mut ss_emitl = vec![-1i32; nnodes + 1];
    let mut ss_emitr = vec![-1i32; nnodes + 1];
    let mut il_used = vec![-1i32; nnodes + 1];
    let mut ir_used = vec![-1i32; nnodes + 1];
    let mut il_ct = vec![0i32; nnodes + 1];
    let mut ir_ct = vec![0i32; nnodes + 1];
    let mut tr_nd_for_bifs = vec![-1i32; nnodes + 1];

    // Walk the sub parsetree, assigning orig states/insert counts (C 3379-3489).
    for x in 0..sub_tr.n as usize {
        let sub_v = sub_tr.state[x];
        let orig_v1 = submap.s2o_smap[sub_v as usize][0];
        let orig_v2 = submap.s2o_smap[sub_v as usize][1];
        if orig_v1 == -1 {
            // S/E/B/EL sub states map to nothing in the orig CM; skip (C 3386-3394).
            continue;
        }
        let orig_nd1 = orig_cm.ndidx[orig_v1 as usize] as usize;
        let st1 = orig_cm.sttype[orig_v1 as usize] as i32;
        let sv1 = st1;
        let sv2 = if orig_v2 != -1 {
            orig_cm.sttype[orig_v2 as usize] as i32
        } else {
            -1
        };
        if st1 == IL_ST {
            il_used[orig_nd1] = orig_v1;
            il_ct[orig_nd1] += 1;
        } else if st1 == IR_ST {
            ir_used[orig_nd1] = orig_v1;
            ir_ct[orig_nd1] += 1;
        } else if sub_cm.ndtype[sub_cm.ndidx[sub_v as usize] as usize] as i32 == MATP_ND {
            ss_used[orig_nd1] = orig_v1;
        } else if orig_cm.ndtype[orig_nd1] as i32 == MATP_ND {
            // sub state maps into an orig MATP node: figure out which split state.
            let base = orig_cm.nodemap[orig_nd1];
            if sub_cm.sttype[sub_v as usize] as i32 == D_ST {
                if ss_used[orig_nd1] == -1
                    || orig_cm.sttype[ss_used[orig_nd1] as usize] as i32 == D_ST
                {
                    ss_used[orig_nd1] = base + 3; // MATP_D
                }
            } else {
                let cur = ss_used[orig_nd1];
                if cur == -1 || orig_cm.sttype[cur as usize] as i32 == D_ST {
                    if sv1 == ML_ST || sv2 == ML_ST {
                        ss_used[orig_nd1] = base + 1; // MATP_ML
                    } else if sv1 == MR_ST || sv2 == MR_ST {
                        ss_used[orig_nd1] = base + 2; // MATP_MR
                    }
                } else if orig_cm.sttype[cur as usize] as i32 == ML_ST {
                    if sv1 == MR_ST || sv2 == MR_ST {
                        ss_used[orig_nd1] = base; // MATP_MP
                    }
                } else if orig_cm.sttype[cur as usize] as i32 == MR_ST {
                    if sv1 == ML_ST || sv2 == ML_ST {
                        ss_used[orig_nd1] = base; // MATP_MP
                    }
                }
            }
        } else {
            ss_used[orig_nd1] = orig_v1;
        }
    }

    // Nodes with no visited state default to D/S/E (non-local) (C 3495-3510).
    for nd in 0..nnodes {
        if ss_used[nd] == -1 {
            let t = orig_cm.ndtype[nd] as i32;
            if t == MATP_ND {
                ss_used[nd] = orig_cm.nodemap[nd] + 3; // MATP_D
            }
            if t == MATL_ND || t == MATR_ND {
                ss_used[nd] = orig_cm.nodemap[nd] + 1; // MAT{L,R}_D
            }
            if t == BIF_ND || t == BEGL_ND || t == BEGR_ND || t == END_ND {
                ss_used[nd] = orig_cm.nodemap[nd]; // BIF_B / BEG{L,R}_S / END_E
            }
        }
    }

    // Determine emitl/emitr for each orig node via the emit-map traversal
    // (C 3517-3564, from CreateEmitMap). Stack entries are pushed as
    // (on_right, state) pairs; state pops first.
    let mut pos = 1i32;
    let mut pda: Vec<i32> = Vec::new();
    pda.push(0); // on_right = 0 (left side)
    pda.push(0); // ss = 0 (ROOT_S state index)
    while let Some(ss) = pda.pop() {
        let on_right = pda.pop().unwrap();
        let nd_ss = orig_cm.ndidx[ss as usize] as usize;
        let st = orig_cm.sttype[ss as usize] as i32;
        if on_right != 0 {
            pos += ir_ct[nd_ss];
            if st == MP_ST || st == MR_ST {
                pos += 1;
            }
            ss_emitr[nd_ss] = pos - 1;
        } else {
            ss_emitl[nd_ss] = pos;
            if st == MP_ST || st == ML_ST {
                pos += 1;
            }
            if st == B_ST {
                // Push BIF's right side, then right child (BEGR), then left (BEGL).
                pda.push(1);
                pda.push(ss);
                pda.push(0);
                pda.push(orig_cm.cnum[ss as usize]); // right child state
                pda.push(0);
                pda.push(orig_cm.cfirst[ss as usize]); // left child state
            } else {
                pda.push(1);
                pda.push(ss);
                if st != E_ST {
                    pda.push(0);
                    pda.push(ss_used[nd_ss + 1]); // split state of child node
                }
            }
            pos += il_ct[nd_ss];
        }
    }

    // Build the orig-CM parsetree in node order (C 3581-3630).
    let mut orig_tr = Parsetree::new(100);
    for cm_nd in 0..nnodes {
        let su = ss_used[cm_nd];
        let st = orig_cm.sttype[su as usize] as i32;
        let emitl_flag = if st == MP_ST || st == ML_ST { 1 } else { 0 };
        let emitr_flag = if st == MP_ST || st == MR_ST { 1 } else { 0 };

        if orig_cm.ndtype[cm_nd] as i32 == BEGR_ND {
            // Attach to the BIF parent's tr node; fix its nxtr afterward.
            let bif_state = orig_cm.plast[orig_cm.nodemap[cm_nd] as usize];
            let parent_tr_nd = tr_nd_for_bifs[orig_cm.ndidx[bif_state as usize] as usize];
            insert_trace_node(
                &mut orig_tr, parent_tr_nd, false, ss_emitl[cm_nd], ss_emitr[cm_nd], su,
            );
            orig_tr.nxtr[parent_tr_nd as usize] = orig_tr.n - 1;
        } else {
            let parent = orig_tr.n - 1;
            insert_trace_node(
                &mut orig_tr, parent, true, ss_emitl[cm_nd], ss_emitr[cm_nd], su,
            );
        }

        if orig_cm.ndtype[cm_nd] as i32 == BIF_ND {
            tr_nd_for_bifs[cm_nd] = orig_tr.n - 1;
        }

        // Left inserts, then right inserts (C 3618-3628).
        for i in 0..il_ct[cm_nd] {
            let parent = orig_tr.n - 1;
            insert_trace_node(
                &mut orig_tr,
                parent,
                true,
                ss_emitl[cm_nd] + emitl_flag + i,
                ss_emitr[cm_nd] - emitr_flag,
                il_used[cm_nd],
            );
        }
        for i in 0..ir_ct[cm_nd] {
            let parent = orig_tr.n - 1;
            insert_trace_node(
                &mut orig_tr,
                parent,
                true,
                ss_emitl[cm_nd] + emitl_flag + il_ct[cm_nd],
                ss_emitr[cm_nd] - emitr_flag - i,
                ir_used[cm_nd],
            );
        }
    }

    orig_tr
}
