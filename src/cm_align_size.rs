// SPDX-License-Identifier: BSD-3-Clause
//! DP matrix size estimation for the cmalign score-report "mem (Mb)" column.
//!
//! C: `cm_alndata.c:DispatchSqAlignment` fills `data->mb_tot` from
//! `cm_*AlignSizeNeeded*()` (the last `&mb_tot` == `ret_totmb` argument), then
//! `output_scores()` (cmalign.c:2040) prints it with `%8.2f`. These sizes are
//! DETERMINISTIC — a pure function of the CM, sequence length `L`, and the HMM
//! bands `cp9b` — NOT timing. So the "mem (Mb)" column MUST match C byte-for-byte
//! (unlike the three "running time (s)" columns, which are the only
//! non-reproducible timing columns).
//!
//! Struct sizes (x86-64, verified against the C headers via `sizeof`):
//!   `CM_HB_MX`=64  `CM_TR_HB_MX`=192  `CM_HB_SHADOW_MX`=104  `CM_TR_HB_SHADOW_MX`=320
//!   `CM_HB_EMIT_MX`=96  `CM_TR_HB_EMIT_MX`=128  `CP9_MX`=112
//!   sizeof(float*)=sizeof(int*)=sizeof(char*)=sizeof(int**)=sizeof(char**)=8
//!   sizeof(int)=sizeof(float)=4  sizeof(char)=1
//!
//! All accumulation is in f32 (C's `float Mb_needed`); the final `*= 0.000001` /
//! `/= 1000000.` is a *double* multiply/divide of the f32 value re-narrowed to f32
//! (C promotes float->double for the double literal, then assigns back to float).
//! Reproduced exactly with `(mb as f64 * K) as f32`.

use crate::cm::CM;
use crate::constants::{B_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST};
use crate::cp9::CP9Bands;

/// C CMH_LOCAL_END (infernal.h:1938); the alignment config sets this bit (1<<11).
const SZ_CMH_LOCAL_END: u32 = 1 << 11;

/// Sum of banded d-widths over all j rows of state v:
/// `sum_{jp=0..jbw} hdmax[v][jp]-hdmin[v][jp]+1`.
#[inline]
fn hd_ncells(cp9b: &CP9Bands, v: usize) -> i64 {
    // C: `for(jp = 0; jp <= jbw; jp++)`. jbw == jmax[v]-jmin[v] may be -1 for an
    // unreachable state (empty band); the i32 range `0..=jbw` is then empty (do NOT
    // cast jbw to usize — that would wrap -1 to a huge range and index the empty row).
    let jbw = cp9b.jmax[v] - cp9b.jmin[v];
    let mut n: i64 = 0;
    let mut jp: i32 = 0;
    while jp <= jbw {
        n += (cp9b.hdmax[v][jp as usize] - cp9b.hdmin[v][jp as usize] + 1) as i64;
        jp += 1;
    }
    n
}

// NOTE: C's cm_*AlignSizeNeeded*HB also computes `cp9mxmb = 2*SizeNeededCP9Matrix(L,
// cm->cp9->M)` and returns `totmb = cmtotmb + cp9mxmb` in ret_totmb. But the reference
// cmalign BINARY prints `cmtotmb` (CM DP matrices only) in the "mem (Mb)" column, so
// the CP9 fwd/bck matrix term is intentionally NOT added below (see wrappers).

/// C `cm_hb_mx_SizeNeeded` (cm_mx.c:1146) — HMM-banded standard CYK/Inside matrix.
fn sz_cm_hb_mx(cm: &CM, cp9b: &CP9Bands, l: i32) -> f32 {
    let have_el = (cm.flags & SZ_CMH_LOCAL_END) != 0;
    let mut ncells: i64 = 0;
    // Mb_needed = sizeof(CM_HB_MX) + (cm_M+1)*sizeof(float**) + (cm_M+1)*sizeof(int)
    let mut mb: f32 =
        (64_i64 + (cp9b.cm_m as i64 + 1) * 8 + (cp9b.cm_m as i64 + 1) * 4) as f32;
    for v in 0..cp9b.cm_m as usize {
        let jbw = cp9b.jmax[v] - cp9b.jmin[v];
        mb += (8_i64 * (jbw as i64 + 1)) as f32; // mx->dp[v][] ptrs
        ncells += hd_ncells(cp9b, v);
    }
    if have_el {
        // ncells += (int64)((int64)(L+2)*(int64)(L+1)*0.5); EL deck
        ncells += (((l as i64 + 2) * (l as i64 + 1)) as f64 * 0.5) as i64;
    }
    mb += (4_i64 * ncells) as f32; // sizeof(float)*ncells  mx->dp_mem
    (mb as f64 * 0.000001) as f32
}

/// C `cm_tr_hb_mx_SizeNeeded` (cm_mx.c:1759) — HMM-banded truncated CYK/Inside matrix.
fn sz_cm_tr_hb_mx(cm: &CM, cp9b: &CP9Bands, l: i32) -> f32 {
    let have_el = (cm.flags & SZ_CMH_LOCAL_END) != 0;
    let (mut jn, mut ln, mut rn, mut tn): (i64, i64, i64, i64) = (0, 0, 0, 0);
    // Mb = sizeof(CM_TR_HB_MX) + 4*(cm_M+1)*sizeof(float**) + 4*(cm_M+1)*sizeof(int)
    let mut mb: f32 =
        (192_i64 + 4 * (cp9b.cm_m as i64 + 1) * 8 + 4 * (cp9b.cm_m as i64 + 1) * 4) as f32;
    for v in 0..cp9b.cm_m as usize {
        let jbw = cp9b.jmax[v] - cp9b.jmin[v];
        if cp9b.jvalid[v] {
            mb += (8_i64 * (jbw as i64 + 1)) as f32;
            jn += hd_ncells(cp9b, v);
        }
        if cp9b.lvalid[v] {
            mb += (8_i64 * (jbw as i64 + 1)) as f32;
            ln += hd_ncells(cp9b, v);
        }
        if cp9b.rvalid[v] {
            mb += (8_i64 * (jbw as i64 + 1)) as f32;
            rn += hd_ncells(cp9b, v);
        }
        if cp9b.tvalid[v] {
            mb += (8_i64 * (jbw as i64 + 1)) as f32;
            tn += hd_ncells(cp9b, v);
        }
    }
    if have_el {
        let el = (((l as i64 + 2) * (l as i64 + 1)) as f64 * 0.5) as i64;
        let m = cp9b.cm_m as usize;
        if cp9b.jvalid[m] {
            mb += (8_i64 * (l as i64 + 1)) as f32;
            jn += el;
        }
        if cp9b.lvalid[m] {
            mb += (8_i64 * (l as i64 + 1)) as f32;
            ln += el;
        }
        if cp9b.rvalid[m] {
            mb += (8_i64 * (l as i64 + 1)) as f32;
            rn += el;
        }
    }
    mb += (4_i64 * jn) as f32; // Jdp_mem
    mb += (4_i64 * ln) as f32; // Ldp_mem
    mb += (4_i64 * rn) as f32; // Rdp_mem
    mb += (4_i64 * tn) as f32; // Tdp_mem
    (mb as f64 * 0.000001) as f32
}

/// C `cm_hb_shadow_mx_SizeNeeded` (cm_mx.c:3187).
fn sz_cm_hb_shadow_mx(cm: &CM, cp9b: &CP9Bands) -> f32 {
    let (mut yn, mut kn): (i64, i64) = (0, 0);
    // Mb = sizeof(CM_HB_SHADOW_MX) + cm_M*sizeof(char**) + cm_M*sizeof(int**) + cm_M*sizeof(int)
    let mut mb: f32 =
        (104_i64 + cp9b.cm_m as i64 * 8 + cp9b.cm_m as i64 * 8 + cp9b.cm_m as i64 * 4) as f32;
    for v in 0..cp9b.cm_m as usize {
        let jbw = cp9b.jmax[v] - cp9b.jmin[v];
        // sizeof(int*)==sizeof(char*)==8, so ptr term is the same either branch
        mb += (8_i64 * (jbw as i64 + 1)) as f32;
        if cm.sttype[v] as i32 == B_ST {
            kn += hd_ncells(cp9b, v);
        } else {
            yn += hd_ncells(cp9b, v);
        }
    }
    mb += yn as f32; // sizeof(char)*y_ncells
    mb += (4_i64 * kn) as f32; // sizeof(int)*k_ncells
    (mb as f64 * 0.000001) as f32
}

/// C `cm_tr_hb_shadow_mx_SizeNeeded` (cm_mx.c:3975).
fn sz_cm_tr_hb_shadow_mx(cm: &CM, cp9b: &CP9Bands) -> f32 {
    let (mut jy, mut ly, mut ry): (i64, i64, i64) = (0, 0, 0);
    let (mut jk, mut lk, mut rk, mut tk): (i64, i64, i64, i64) = (0, 0, 0, 0);
    // Mb = sizeof(CM_TR_HB_SHADOW_MX) + 3*cm_M*sizeof(char**) + 4*cm_M*sizeof(int**) + 4*cm_M*sizeof(int)
    let mut mb: f32 =
        (320_i64 + 3 * cp9b.cm_m as i64 * 8 + 4 * cp9b.cm_m as i64 * 8 + 4 * cp9b.cm_m as i64 * 4)
            as f32;
    for v in 0..cp9b.cm_m as usize {
        let jbw = cp9b.jmax[v] - cp9b.jmin[v];
        if cm.sttype[v] as i32 == B_ST {
            if cp9b.jvalid[v] {
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Jkshadow[v][] ptrs
                jk += hd_ncells(cp9b, v);
            }
            if cp9b.lvalid[v] {
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Lkshadow[v][] ptrs
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Lkmode[v][] ptrs
                lk += hd_ncells(cp9b, v);
            }
            if cp9b.rvalid[v] {
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Rkshadow[v][] ptrs
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Rkmode[v][] ptrs
                rk += hd_ncells(cp9b, v);
            }
            if cp9b.tvalid[v] {
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Tkshadow[v][] ptrs
                tk += hd_ncells(cp9b, v);
            }
        } else {
            if cp9b.jvalid[v] {
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Jyshadow[v][] ptrs
                jy += hd_ncells(cp9b, v);
            }
            if cp9b.lvalid[v] {
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Lyshadow[v][] ptrs
                ly += hd_ncells(cp9b, v);
            }
            if cp9b.rvalid[v] {
                mb += (8_i64 * (jbw as i64 + 1)) as f32; // Ryshadow[v][] ptrs
                ry += hd_ncells(cp9b, v);
            }
        }
    }
    mb += (4_i64 * jk) as f32; // Jkshadow_mem (int)
    mb += (4_i64 * lk) as f32; // Lkshadow_mem
    mb += (4_i64 * rk) as f32; // Rkshadow_mem
    mb += (4_i64 * tk) as f32; // Tkshadow_mem
    mb += lk as f32; // Lkmode_mem (char)
    mb += rk as f32; // Rkmode_mem
    mb += jy as f32; // Jyshadow_mem (char)
    mb += ly as f32; // Lyshadow_mem
    mb += ry as f32; // Ryshadow_mem
    (mb as f64 * 0.000001) as f32
}

/// C `cm_hb_emit_mx_SizeNeeded` (cm_mx.c:5196).
fn sz_cm_hb_emit_mx(cm: &CM, cp9b: &CP9Bands, l: i32) -> f32 {
    let have_el = (cm.flags & SZ_CMH_LOCAL_END) != 0;
    let (mut ln, mut rn): (i64, i64) = (0, 0);
    // Mb = sizeof(CM_HB_EMIT_MX) + (cm->M+1)*sizeof(float*) [l_pp] + (cm->M+1)*sizeof(float*) [r_pp]
    let mut mb: f32 = (96_i64 + (cm.m as i64 + 1) * 8 + (cm.m as i64 + 1) * 8) as f32;
    for v in 0..cm.m as usize {
        let st = cm.sttype[v] as i32;
        if st == MP_ST || st == ML_ST || st == IL_ST {
            ln += (cp9b.imax[v] - cp9b.imin[v] + 1) as i64;
        }
        if st == MP_ST || st == MR_ST || st == IR_ST {
            rn += (cp9b.jmax[v] - cp9b.jmin[v] + 1) as i64;
        }
    }
    if have_el {
        ln += l as i64 + 1;
    }
    // sizeof(float)*(l_ncells + r_ncells + (L+1))  [l_pp_mem, r_pp_mem, sum]
    mb += (4_i64 * (ln + rn + (l as i64 + 1))) as f32;
    (mb as f64 * 0.000001) as f32
}

/// C `cm_tr_hb_emit_mx_SizeNeeded` (cm_mx.c:5627).
fn sz_cm_tr_hb_emit_mx(cm: &CM, cp9b: &CP9Bands, l: i32) -> f32 {
    let have_el = (cm.flags & SZ_CMH_LOCAL_END) != 0;
    let (mut ln, mut rn): (i64, i64) = (0, 0);
    // Mb = sizeof(CM_TR_HB_EMIT_MX) + 4*(cm->M+1)*sizeof(float*)  [Jl,Ll,Jr,Rr]
    let mut mb: f32 = (128_i64 + 4 * (cm.m as i64 + 1) * 8) as f32;
    for v in 0..cm.m as usize {
        let st = cm.sttype[v] as i32;
        if st == MP_ST || st == ML_ST || st == IL_ST {
            ln += (cp9b.imax[v] - cp9b.imin[v] + 1) as i64;
        }
        if st == MP_ST || st == MR_ST || st == IR_ST {
            rn += (cp9b.jmax[v] - cp9b.jmin[v] + 1) as i64;
        }
    }
    if have_el {
        ln += l as i64 + 1;
        rn += l as i64 + 1;
    }
    // sizeof(float)*(l+l+r+r+(L+1))  [Jl_pp_mem, Ll_pp_mem, Jr_pp_mem, Rr_pp_mem, sum]
    mb += (4_i64 * (ln + ln + rn + rn + (l as i64 + 1))) as f32;
    (mb as f64 * 0.000001) as f32
}

/// C `cm_TrAlignSizeNeededHB` (cm_dpalign_trunc.c:905) returning `totmb` — the
/// default cmalign truncated HMM-banded path. `do_post` == want_pp; `do_sample`
/// is false on the alignment (non-sampling) path.
pub fn cm_tr_align_size_needed_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    _cp9_m: i32,
    l: i32,
    do_post: bool,
    do_sample: bool,
) -> f32 {
    let mxmb = sz_cm_tr_hb_mx(cm, cp9b, l);
    let mut cmtotmb = mxmb;
    if do_post {
        cmtotmb += mxmb; // Outside/Posterior matrix (reused, counted once)
        cmtotmb += sz_cm_tr_hb_emit_mx(cm, cp9b, l);
    }
    if !do_sample {
        cmtotmb += sz_cm_tr_hb_shadow_mx(cm, cp9b);
    }
    // C reports data->mb_tot == cmtotmb (the CM DP matrices only). The reference
    // binary's "mem (Mb)" column is cmtotmb, NOT cmtotmb + 2*SizeNeededCP9Matrix:
    // verified byte-exact across all fixture rows (e.g. L=72 -> cmtotmb=0.3609 ->
    // "0.36", matching C; adding the ~0.176 Mb of CP9 fwd/bck matrices would give
    // "0.54", which C does NOT print). So the CP9-matrix term is excluded here.
    cmtotmb
}

/// C `cm_AlignSizeNeededHB` (cm_dpalign.c:523) returning `totmb` — the standard
/// (`--notrunc`) HMM-banded path.
pub fn cm_align_size_needed_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    _cp9_m: i32,
    l: i32,
    do_post: bool,
    do_sample: bool,
) -> f32 {
    let mxmb = sz_cm_hb_mx(cm, cp9b, l);
    let mut cmtotmb = mxmb;
    if do_post {
        cmtotmb += mxmb;
        cmtotmb += sz_cm_hb_emit_mx(cm, cp9b, l);
    }
    if !do_sample {
        cmtotmb += sz_cm_hb_shadow_mx(cm, cp9b);
    }
    // As above: the reference binary's mem column is cmtotmb (CM matrices only),
    // excluding the 2*SizeNeededCP9Matrix CP9 fwd/bck term.
    cmtotmb
}
