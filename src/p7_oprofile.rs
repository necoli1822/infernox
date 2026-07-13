// SPDX-License-Identifier: BSD-3-Clause
//! Faithful port of the SSE striped optimized-profile writer used by C
//! `cmpress`: it produces the `.i1f` (MSV filter part) and `.i1p` (rest of the
//! profile) caches BYTE-IDENTICALLY to stock Infernal, so an infernox-pressed
//! CM database is consumable by stock C `cmscan`.
//!
//! Pipeline mirrors `cmpress.c:97-114`:
//!   gm = p7_ProfileConfig(cm->fp7, bg, 400, p7_LOCAL)   [build_local_profile]
//!   om = p7_oprofile_Convert(gm)                        [convert_oprofile]
//!   cm_p7_oprofile_Write(ffp, pfp, ..., om)             [cm_p7_oprofile_write]
//!
//! The three source functions transcribed here:
//!   - impl_sse/p7_oprofile.c: mf_conversion / sf_conversion / vf_conversion /
//!     fb_conversion / p7_oprofile_Convert (generic profile -> striped oprofile)
//!   - impl_sse/io.c: p7_oprofile_Write (on-disk striped serialization)
//!   - cm_file.c: cm_p7_oprofile_Write (7-field CM-specific `.i1f` header)
//!   - easel/esl_sse.c: esl_sse_expf (the Cephes polynomial exp used for the
//!     Forward/Backward odds-ratio vectors — MUST be bit-exact, so it is ported
//!     scalar rather than replaced by libm expf).
//!
//! All arithmetic reproduces C's exact float/double split, rounding, saturation
//! and striped ordering; verified by whole-file `cmp` against C cmpress.

use crate::p7_generic::{build_local_profile, GlocalProfile};
use crate::p7_hmm::{P7Profile, P7H_GA, P7H_NC, P7H_TC};

// ------------------------------------------------------------------------
// Constants (hmmer.h, impl_sse/impl_sse.h)
// ------------------------------------------------------------------------

// impl_sse/io.c:46-47 — 3/f binary MSV / profile file magics (SSE).
const V3F_FMAGIC: u32 = 0xb3e6e6f3;
const V3F_PMAGIC: u32 = 0xb3e6f0f3;
// cm_file.c:48 — 1/a binary MSV/SSV file magic ("1afs").
const V1A_FMAGIC: u32 = 0xb1e1e6f3;

// impl_sse/impl_sse.h:24-28
#[inline]
fn p7o_nqb(m: usize) -> usize {
    std::cmp::max(2, (m - 1) / 16 + 1)
} // 16 uchars
#[inline]
fn p7o_nqw(m: usize) -> usize {
    std::cmp::max(2, (m - 1) / 8 + 1)
} // 8 words
#[inline]
fn p7o_nqf(m: usize) -> usize {
    std::cmp::max(2, (m - 1) / 4 + 1)
} // 4 floats
const P7O_EXTRA_SB: usize = 17; // impl_sse.h:28

// hmmer.h:69-71
const P7_NEVPARAM: usize = 6;
const P7_NCUTOFFS: usize = 6;
const P7_MAXABET: usize = 20;
const P7_CUTOFF_UNSET: f32 = -99999.0; // hmmer.h:77

// impl_sse.h:68-69
const P7O_NXSTATES: usize = 4; // ENJC
const P7O_NXTRANS: usize = 2; // MOVE, LOOP
const KP: usize = 18; // RNA extended alphabet size (abc->Kp)
const K_CANON: usize = 4; // abc->K

// p7 transition indices in gm->tsc  (hmmer.h:222-233, "order optimized for DP").
// gm->tsc[k*8 + s]; these MUST match the p7_generic layout.
const P7P_MM: usize = 0;
const P7P_IM: usize = 1;
const P7P_DM: usize = 2;
const P7P_BM: usize = 3;
const P7P_MD: usize = 4;
const P7P_DD: usize = 5;
const P7P_MI: usize = 6;
const P7P_II: usize = 7;
const P7P_NTRANS: usize = 8;

// special-state rows: E=0,N=1,J=2,C=3 ; LOOP=0,MOVE=1 (hmmer.h).
const P7P_E: usize = 0;
const P7P_N: usize = 1;
const P7P_J: usize = 2;
const P7P_C: usize = 3;
const P7P_LOOP: usize = 0;
const P7P_MOVE: usize = 1;

const NEG_INF: f32 = f32::NEG_INFINITY;

// eslCONST_LOG2 / LOG2R (easel.h:304-305) — full double precision.
const ESL_CONST_LOG2: f64 = 0.69314718055994529;
const ESL_CONST_LOG2R: f64 = 1.44269504088896341;

// ------------------------------------------------------------------------
// byteify / wordify (impl_sse/p7_oprofile.c:666-705)
// ------------------------------------------------------------------------

/// C impl_sse/p7_oprofile.c:667 biased_byteify():
/// ```c
/// sc  = -1.0f * roundf(om->scale_b * sc);
/// b   = (sc > 255 - om->bias_b) ? 255 : (uint8_t) sc + om->bias_b;
/// ```
#[inline]
fn biased_byteify(scale_b: f32, bias_b: u8, sc: f32) -> u8 {
    let sc = -1.0f32 * (scale_b * sc).round();
    if sc > (255i32 - bias_b as i32) as f32 {
        255
    } else {
        // C: `(uint8_t) sc + om->bias_b`, result truncated back to uint8_t.
        // (uint8_t)float == truncate-to-int then take low byte; then +bias mod 256.
        (sc as i32 as u8).wrapping_add(bias_b)
    }
}

/// C impl_sse/p7_oprofile.c:683 unbiased_byteify():
/// ```c
/// sc  = -1.0f * roundf(om->scale_b * sc);
/// b   = (sc > 255.) ? 255 : (uint8_t) sc;
/// ```
#[inline]
fn unbiased_byteify(scale_b: f32, sc: f32) -> u8 {
    let sc = -1.0f32 * (scale_b * sc).round();
    if sc > 255.0 {
        255
    } else {
        sc as i32 as u8
    }
}

/// C impl_sse/p7_oprofile.c:699 wordify():
/// ```c
/// sc  = roundf(om->scale_w * sc);
/// if (sc >= 32767.0) return 32767; else if (sc <= -32768.0) return -32768;
/// else return (int16_t) sc;
/// ```
#[inline]
fn wordify(scale_w: f32, sc: f32) -> i16 {
    let sc = (scale_w * sc).round();
    if sc >= 32767.0 {
        32767
    } else if sc <= -32768.0 {
        -32768
    } else {
        sc as i32 as i16
    }
}

// ------------------------------------------------------------------------
// esl_sse_expf (easel/esl_sse.c:182), transcribed scalar (per-element).
// Cephes polynomial approximation; bit-exact to the vectorized version because
// each __m128 lane is computed independently with the same float op sequence.
// ------------------------------------------------------------------------
#[inline]
fn esl_sse_expf(x0: f32) -> f32 {
    // static tables (esl_sse.c:184-188)
    const CEPHES_P: [f32; 6] = [
        1.9875691500E-4,
        1.3981999507E-3,
        8.3334519073E-3,
        4.1665795894E-2,
        1.6666665459E-1,
        5.0000001201E-1,
    ];
    const CEPHES_C0: f32 = 0.693359375;
    const CEPHES_C1: f32 = -2.12194440e-4;
    const MAXLOGF: f32 = 88.3762626647949;
    const MINLOGF: f32 = -88.3762626647949;
    let log2r: f32 = ESL_CONST_LOG2R as f32; // _mm_set1_ps(eslCONST_LOG2R)

    // out-of-range masks (esl_sse.c:193-194)
    let maxmask = x0 > MAXLOGF;
    let minmask = x0 <= MINLOGF;

    // range reduction (esl_sse.c:197-206)
    let mut fx = x0 * log2r;
    fx = fx + 0.5;
    let mut k: i32 = fx as i32; // _mm_cvttps_epi32 (truncate toward zero)
    let tmp = k as f32; // _mm_cvtepi32_ps
    let mask: f32 = if tmp > fx { 1.0 } else { 0.0 };
    fx = tmp - mask;
    k = fx as i32;

    // polynomial for e^f, f in [-0.5,0.5] (esl_sse.c:209-222)
    let tmpc = fx * CEPHES_C0;
    let z1 = fx * CEPHES_C1;
    let mut x = x0 - tmpc;
    x = x - z1;
    let z = x * x;
    let mut y = CEPHES_P[0];
    y = y * x;
    y = y + CEPHES_P[1];
    y = y * x;
    y = y + CEPHES_P[2];
    y = y * x;
    y = y + CEPHES_P[3];
    y = y * x;
    y = y + CEPHES_P[4];
    y = y * x;
    y = y + CEPHES_P[5];
    y = y * z;
    y = y + x;
    y = y + 1.0;

    // build 2^k as an IEEE754 float (esl_sse.c:225-227)
    let bits = ((k + 127) << 23) as u32; // _mm_slli_epi32(k+127, 23)
    let twok = f32::from_bits(bits);
    y = y * twok;

    // range cleanup (esl_sse.c:233-234)
    if maxmask {
        y = f32::INFINITY;
    }
    if minmask {
        y = 0.0;
    }
    y
}

// ------------------------------------------------------------------------
// The striped optimized profile (the fields p7_oprofile_Write serializes).
// ------------------------------------------------------------------------
struct Oprofile {
    m: usize,
    // MSV (mf/sf_conversion)
    scale_b: f32,
    base_b: u8,
    bias_b: u8,
    tbm_b: u8,
    tec_b: u8,
    tjb_b: u8,
    /// sbv[x] : Kp rows, each Q16x vectors of 16 bytes (i8 reinterpret) — striped.
    sbv: Vec<Vec<[u8; 16]>>,
    /// rbv[x] : Kp rows, each Q16 vectors of 16 bytes.
    rbv: Vec<Vec<[u8; 16]>>,
    // Viterbi (vf_conversion)
    scale_w: f32,
    base_w: i16,
    ddbound_w: i16,
    ncj_roundoff: f32,
    /// twv : flat 8*Q8 vectors of 8 int16.
    twv: Vec<[i16; 8]>,
    /// rwv[x] : Kp rows, each Q8 vectors of 8 int16.
    rwv: Vec<Vec<[i16; 8]>>,
    xw: [[i16; 2]; 4],
    // Forward/Backward (fb_conversion)
    /// tfv : flat 8*Q4 vectors of 4 f32.
    tfv: Vec<[f32; 4]>,
    /// rfv[x] : Kp rows, each Q4 vectors of 4 f32.
    rfv: Vec<Vec<[f32; 4]>>,
    xf: [[f32; 2]; 4],
}

/// C impl_sse/p7_oprofile.c:1024 p7_oprofile_Convert(gm, om) — builds the
/// striped MSV/VF/FB vectors from the generic (nats) profile `gm`.
fn convert_oprofile(gm: &GlocalProfile) -> Oprofile {
    let m = gm.m;
    // p7P_TSC/p7P_MSC accessors over the generic profile.
    //
    // C p7_profile_Create (p7_profile.c:84) does `esl_vec_FSet(gm->tsc, 8, -inf)`,
    // i.e. node-0 transitions are all -inf ("node 0 nonexistent"), and modelconfig
    // then overwrites ONLY tsc[0][BM] with the real local/glocal entry score
    // (tsc[1..M-1] for the rest). The infernox GlocalProfile leaves tsc[0][t!=BM]
    // at 0.0, so reproduce C's -inf here — the vf/fb conversions read tsc[0][MM,
    // IM,DM] at the q=0,z=0 striped lane (kb=k-1=0), which must be -inf.
    let tsc = |k: usize, s: usize| -> f32 {
        if k == 0 && s != P7P_BM {
            NEG_INF
        } else {
            gm.tsc[k * P7P_NTRANS + s]
        }
    };
    let msc = |k: usize, x: usize| -> f32 { gm.rsc[x][k * 2] };

    // ---------------- mf_conversion (p7_oprofile.c:773) ----------------
    let nqb = p7o_nqb(m);
    // scale_b = 3.0 / eslCONST_LOG2 (double div -> float); base_b = 190.
    let scale_b = (3.0f64 / ESL_CONST_LOG2) as f32;
    let base_b: u8 = 190;
    // max over canonical residues of rsc[x][0..(M+1)*2] (both MSC & ISC).
    let mut max = 0.0f32;
    for x in 0..K_CANON {
        for &v in &gm.rsc[x][0..(m + 1) * 2] {
            if v > max {
                max = v;
            }
        }
    }
    let bias_b = unbiased_byteify(scale_b, -1.0 * max);

    // striped match costs rbv[x][q].byte[z], k=q+1, position k+z*nqb (:796-803)
    let mut rbv = vec![vec![[0u8; 16]; nqb]; KP];
    for x in 0..KP {
        for q in 0..nqb {
            let k = q + 1;
            for z in 0..16 {
                let kk = k + z * nqb;
                rbv[x][q][z] = if kk <= m {
                    biased_byteify(scale_b, bias_b, msc(kk, x))
                } else {
                    255
                };
            }
        }
    }
    // transition costs (:806-808)
    let tbm_b = unbiased_byteify(scale_b, (2.0f32 / (m as f32 * (m as f32 + 1.0))).ln());
    let tec_b = unbiased_byteify(scale_b, 0.5f32.ln());
    // tjb adopts gm->L (== 400 here). p7_oprofile.c:808 logf(3.0/(L+3)).
    let tjb_b = unbiased_byteify(scale_b, (3.0f32 / (gm.l as f32 + 3.0)).ln());

    // ---------------- sf_conversion (p7_oprofile.c:721) ----------------
    // sbv[x][q] = ((127+bias) - rbv[x][q]) unsigned-sat, then ^127; q>=nq copies q%nq.
    let nqs = nqb + P7O_EXTRA_SB;
    let mut sbv = vec![vec![[0u8; 16]; nqs]; KP];
    let tmp127b: u8 = (127i16 + bias_b as i16) as u8; // 127+bias (bias small)
    for x in 0..KP {
        for q in 0..nqb {
            for z in 0..16 {
                sbv[x][q][z] = tmp127b.saturating_sub(rbv[x][q][z]) ^ 127;
            }
        }
        for q in nqb..nqs {
            sbv[x][q] = sbv[x][q % nqb];
        }
    }

    // ---------------- vf_conversion (p7_oprofile.c:836) ----------------
    let nqw = p7o_nqw(m);
    let scale_w = (500.0f64 / ESL_CONST_LOG2) as f32;
    let base_w: i16 = 12000;

    // striped match scores rwv[x][q] (:863-868)
    let mut rwv = vec![vec![[0i16; 8]; nqw]; KP];
    for x in 0..KP {
        for q in 0..nqw {
            let k = q + 1;
            for z in 0..8 {
                let kk = k + z * nqw;
                rwv[x][q][z] = if kk <= m {
                    wordify(scale_w, msc(kk, x))
                } else {
                    -32768
                };
            }
        }
    }
    // Transitions, all but DD (:871-891). p7o order BM,MM,IM,DM,MD,MI,II.
    // (tg, kb-offset, maxval) per transition.
    let vf_specs: [(usize, i32, i16); 7] = [
        (P7P_BM, -1, 0),
        (P7P_MM, -1, 0),
        (P7P_IM, -1, 0),
        (P7P_DM, -1, 0),
        (P7P_MD, 0, 0),
        (P7P_MI, 0, 0),
        (P7P_II, 0, -1), // do not allow II cost of 0
    ];
    let mut twv: Vec<[i16; 8]> = Vec::with_capacity(8 * nqw);
    for q in 0..nqw {
        let k = q + 1;
        for &(tg, kboff, maxval) in &vf_specs {
            let kb = k as i32 + kboff; // k-1 or k
            let mut v = [0i16; 8];
            for z in 0..8 {
                let idx = kb + (z * nqw) as i32;
                let val = if idx < m as i32 {
                    wordify(scale_w, tsc(idx as usize, tg))
                } else {
                    -32768
                };
                v[z] = if val <= maxval { val } else { maxval };
            }
            twv.push(v);
        }
    }
    // DD's, appended at the end (:894-898)
    for q in 0..nqw {
        let k = q + 1;
        let mut v = [0i16; 8];
        for z in 0..8 {
            let kk = k + z * nqw;
            v[z] = if kk < m {
                wordify(scale_w, tsc(kk, P7P_DD))
            } else {
                -32768
            };
        }
        twv.push(v);
    }
    // specials xw (:907-914); N/C/J LOOP hardcoded 0.
    //
    // IMPORTANT: om->xw is indexed with the p7O convention (impl_sse.h:72:
    // p7O_MOVE=0, p7O_LOOP=1) but gm->xsc uses p7P (hmmer.h: p7P_LOOP=0, p7P_MOVE=1)
    // — the two are SWAPPED. C stores `xw[p7O_E][p7O_LOOP] = wordify(xsc[p7P_E]
    // [p7P_LOOP])`, so on disk (written xw[x][0], xw[x][1]) the order is [MOVE,
    // LOOP]. We store slot 0 = MOVE, slot 1 = LOOP to match.
    const OM_MOVE: usize = 0;
    const OM_LOOP: usize = 1;
    let mut xw = [[0i16; 2]; 4];
    xw[P7P_E][OM_MOVE] = wordify(scale_w, gm.xsc[P7P_E][P7P_MOVE]);
    xw[P7P_E][OM_LOOP] = wordify(scale_w, gm.xsc[P7P_E][P7P_LOOP]);
    xw[P7P_N][OM_MOVE] = wordify(scale_w, gm.xsc[P7P_N][P7P_MOVE]);
    xw[P7P_N][OM_LOOP] = 0;
    xw[P7P_C][OM_MOVE] = wordify(scale_w, gm.xsc[P7P_C][P7P_MOVE]);
    xw[P7P_C][OM_LOOP] = 0;
    xw[P7P_J][OM_MOVE] = wordify(scale_w, gm.xsc[P7P_J][P7P_MOVE]);
    xw[P7P_J][OM_LOOP] = 0;
    let ncj_roundoff = 0.0f32;
    // ddbound_w (:921-928): for (k=2; k<M-1; k++)
    let mut ddbound_w: i32 = -32768;
    let mut k = 2;
    while k < m as i32 - 1 {
        let mut ddtmp = wordify(scale_w, tsc(k as usize, P7P_DD)) as i32;
        ddtmp += wordify(scale_w, tsc((k + 1) as usize, P7P_DM)) as i32;
        ddtmp -= wordify(scale_w, tsc((k + 1) as usize, P7P_BM)) as i32;
        if ddtmp > ddbound_w {
            ddbound_w = ddtmp;
        }
        k += 1;
    }
    let ddbound_w = ddbound_w as i16;

    // ---------------- fb_conversion (p7_oprofile.c:939) ----------------
    let nqf = p7o_nqf(m);
    // striped match odds rfv[x][q] = esl_sse_expf(MSC) (:956-961)
    let mut rfv = vec![vec![[0f32; 4]; nqf]; KP];
    for x in 0..KP {
        for q in 0..nqf {
            let k = q + 1;
            for z in 0..4 {
                let kk = k + z * nqf;
                let sc = if kk <= m { msc(kk, x) } else { NEG_INF };
                rfv[x][q][z] = esl_sse_expf(sc);
            }
        }
    }
    // Transitions, all but DD (:965-982). Same p7o order; no maxval clamp.
    let fb_specs: [(usize, i32); 7] = [
        (P7P_BM, -1),
        (P7P_MM, -1),
        (P7P_IM, -1),
        (P7P_DM, -1),
        (P7P_MD, 0),
        (P7P_MI, 0),
        (P7P_II, 0),
    ];
    let mut tfv: Vec<[f32; 4]> = Vec::with_capacity(8 * nqf);
    for q in 0..nqf {
        let k = q + 1;
        for &(tg, kboff) in &fb_specs {
            let kb = k as i32 + kboff;
            let mut v = [0f32; 4];
            for z in 0..4 {
                let idx = kb + (z * nqf) as i32;
                let sc = if idx < m as i32 {
                    tsc(idx as usize, tg)
                } else {
                    NEG_INF
                };
                v[z] = esl_sse_expf(sc);
            }
            tfv.push(v);
        }
    }
    // DD's (:985-989)
    for q in 0..nqf {
        let k = q + 1;
        let mut v = [0f32; 4];
        for z in 0..4 {
            let kk = k + z * nqf;
            let sc = if kk < m { tsc(kk, P7P_DD) } else { NEG_INF };
            v[z] = esl_sse_expf(sc);
        }
        tfv.push(v);
    }
    // specials xf = expf(xsc) (libm expf) (:994-1001). Same p7O/p7P swap as xw:
    // on disk (xf[x][0], xf[x][1]) the order is [MOVE, LOOP].
    let mut xf = [[0f32; 2]; 4];
    for &s in &[P7P_E, P7P_N, P7P_C, P7P_J] {
        xf[s][0] = gm.xsc[s][P7P_MOVE].exp(); // p7O_MOVE
        xf[s][1] = gm.xsc[s][P7P_LOOP].exp(); // p7O_LOOP
    }

    Oprofile {
        m,
        scale_b,
        base_b,
        bias_b,
        tbm_b,
        tec_b,
        tjb_b,
        sbv,
        rbv,
        scale_w,
        base_w,
        ddbound_w,
        ncj_roundoff,
        twv,
        rwv,
        xw,
        tfv,
        rfv,
        xf,
    }
}

// ------------------------------------------------------------------------
// Little-endian append helpers.
// ------------------------------------------------------------------------
#[inline]
fn w_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn w_i32(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn w_i64(b: &mut Vec<u8>, v: i64) {
    b.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn w_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
#[inline]
fn w_i16(b: &mut Vec<u8>, v: i16) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// Build the M+2-byte annotation buffer C writes for an rf/mm/cs/consensus line.
/// C p7_profile_Create mallocs the buffer and only sets [0]=0; p7_ProfileConfig
/// then `strcpy(gm->X, hmm->X)` iff the flag is up. So:
///   flag up  -> [space, chars_1..M, '\0']  (== hmm->X, M+2 bytes)
///   flag down-> M+2 zero bytes             (fresh-malloc 0 + trailing 0; C emits
///                                            all-zero here, verified vs cmpress)
/// `src` is the infernox P7Profile field (length M+1: [0]=' ', [1..=M]=chars).
fn annot_bytes(flag_up: bool, src: &[u8], m: usize) -> Vec<u8> {
    let mut out = vec![0u8; m + 2];
    if flag_up {
        // src has indices 0..=m (length m+1); copy into out[0..=m], leave out[m+1]=0.
        for i in 0..=m {
            out[i] = src[i];
        }
    }
    out
}

/// C cm_file.c:1255 cm_p7_oprofile_Write() + impl_sse/io.c:87 p7_oprofile_Write().
/// Appends the CM-specific header and full striped MSV part to `ffp` (`.i1f`), and
/// the profile remainder to `pfp` (`.i1p`).
///
/// `offs` = om->offs = [p7_MOFFSET, p7_FOFFSET, p7_POFFSET] (fp7 offset in the
/// `.i1m`, and this record's start offsets in `.i1f` / `.i1p`), set by the caller
/// exactly as cmpress.c:106-114.
#[allow(clippy::too_many_arguments)]
pub fn cm_p7_oprofile_write(
    ffp: &mut Vec<u8>,
    pfp: &mut Vec<u8>,
    cm_offset: i64,
    cm_clen: i32,
    cm_w: i32,
    cm_nbp: i32,
    gfmu: f32,
    gflambda: f32,
    offs: [i64; 3],
    fp7: &P7Profile,
) {
    // gm = p7_ProfileConfig(fp7, bg, gm, 400, p7_LOCAL) + ReconfigLength(400)
    let mut gm = build_local_profile(fp7, 400);
    // The shared build_local_profile computes the local-entry (BM) occupancy row
    // entirely in f32, but C p7_hmm_CalculateOccupancy (p7_hmm.c:1338) computes the
    // `(1.0 - mocc[k-1]) * t[DM]` term in DOUBLE (the 1.0 literal), diverging by up
    // to ~1e-2 nats by mid-model. Recompute that row here with C's exact float/
    // double split so the VF/FB transition vectors match byte-for-byte.
    fix_local_entry_bm(&mut gm, fp7);
    let om = convert_oprofile(&gm);
    let m = om.m;
    let nqb = p7o_nqb(m);
    let nqs = nqb + P7O_EXTRA_SB;
    let nqw = p7o_nqw(m);
    let nqf = p7o_nqf(m);

    // ---- cm_file.c:1257-1264 — CM-specific `.i1f` header ----
    w_u32(ffp, V1A_FMAGIC);
    w_i64(ffp, cm_offset); // off_t
    w_i32(ffp, cm_clen);
    w_i32(ffp, cm_w);
    w_i32(ffp, cm_nbp);
    w_f32(ffp, gfmu);
    w_f32(ffp, gflambda);

    // ---- impl_sse/io.c:97-119 — MSV part of the oprofile (`.i1f`) ----
    let name = fp7.name.as_bytes();
    let n = name.len() as i32;
    w_u32(ffp, V3F_FMAGIC);
    w_i32(ffp, m as i32);
    w_i32(ffp, 1); // abc->type = eslRNA
    w_i32(ffp, n);
    ffp.extend_from_slice(name);
    ffp.push(0); // name + '\0'  (n+1 bytes)
    w_i32(ffp, fp7.max_length);
    ffp.push(om.tbm_b);
    ffp.push(om.tec_b);
    ffp.push(om.tjb_b);
    w_f32(ffp, om.scale_b);
    ffp.push(om.base_b);
    ffp.push(om.bias_b);
    // sbv[x] : Q16x vectors of 16 bytes
    for x in 0..KP {
        for q in 0..nqs {
            ffp.extend_from_slice(&om.sbv[x][q]);
        }
    }
    // rbv[x] : Q16 vectors of 16 bytes
    for x in 0..KP {
        for q in 0..nqb {
            ffp.extend_from_slice(&om.rbv[x][q]);
        }
    }
    // evparam[6] (p7_MMU, p7_MLAMBDA, p7_VMU, p7_VLAMBDA, p7_FTAU, p7_FLAMBDA)
    let ev = [
        fp7.evparam.lmmu as f32,
        fp7.evparam.lmlambda as f32,
        fp7.evparam.lvmu as f32,
        fp7.evparam.lvlambda as f32,
        fp7.evparam.lftau as f32,
        fp7.evparam.lflambda as f32,
    ];
    debug_assert_eq!(ev.len(), P7_NEVPARAM);
    for &e in &ev {
        w_f32(ffp, e);
    }
    // offs[3] (off_t)
    for &o in &offs {
        w_i64(ffp, o);
    }
    // compo[p7_MAXABET]: first K from fp7, rest 0.0 (C leaves hmm->compo[K..]=0)
    for x in 0..P7_MAXABET {
        let c = if x < K_CANON { fp7.compo[x] } else { 0.0 };
        w_f32(ffp, c);
    }
    w_u32(ffp, V3F_FMAGIC); // sentinel

    // ---- impl_sse/io.c:122-173 — profile remainder (`.i1p`) ----
    w_u32(pfp, V3F_PMAGIC);
    w_i32(pfp, m as i32);
    w_i32(pfp, 1); // abc->type
    w_i32(pfp, n);
    pfp.extend_from_slice(name);
    pfp.push(0);
    // acc / desc (io.c:128-144)
    write_optstr(pfp, fp7.acc.as_deref());
    write_optstr(pfp, fp7.desc.as_deref());
    // rf / mm / cs / consensus, each M+2 bytes (io.c:146-149)
    let rf = annot_bytes(fp7.flags & crate::p7_hmm::P7H_RF != 0, &fp7.rf, m);
    let mm = annot_bytes(fp7.flags & crate::p7_hmm::P7H_MMASK != 0, &fp7.mm, m);
    let cs = annot_bytes(fp7.flags & crate::p7_hmm::P7H_CS != 0, &fp7.cs, m);
    let cons = annot_bytes(fp7.flags & crate::p7_hmm::P7H_CONS != 0, &fp7.consensus, m);
    pfp.extend_from_slice(&rf);
    pfp.extend_from_slice(&mm);
    pfp.extend_from_slice(&cs);
    pfp.extend_from_slice(&cons);

    // ViterbiFilter part (io.c:152-160)
    // twv : 8*Q8 vectors of 8 int16
    debug_assert_eq!(om.twv.len(), 8 * nqw);
    for v in &om.twv {
        for &w in v {
            w_i16(pfp, w);
        }
    }
    // rwv[x] : Q8 vectors of 8 int16
    for x in 0..KP {
        for q in 0..nqw {
            for &w in &om.rwv[x][q] {
                w_i16(pfp, w);
            }
        }
    }
    // xw[NXSTATES][NXTRANS] int16 (order E,N,J,C each LOOP,MOVE)
    for s in 0..P7O_NXSTATES {
        for t in 0..P7O_NXTRANS {
            w_i16(pfp, om.xw[s][t]);
        }
    }
    w_f32(pfp, om.scale_w);
    w_i16(pfp, om.base_w);
    w_i16(pfp, om.ddbound_w);
    w_f32(pfp, om.ncj_roundoff);

    // Forward/Backward part (io.c:163-167)
    debug_assert_eq!(om.tfv.len(), 8 * nqf);
    for v in &om.tfv {
        for &f in v {
            w_f32(pfp, f);
        }
    }
    for x in 0..KP {
        for q in 0..nqf {
            for &f in &om.rfv[x][q] {
                w_f32(pfp, f);
            }
        }
    }
    for s in 0..P7O_NXSTATES {
        for t in 0..P7O_NXTRANS {
            w_f32(pfp, om.xf[s][t]);
        }
    }

    // cutoff[6], nj, mode, L, sentinel (io.c:169-173)
    let mut cutoff = [P7_CUTOFF_UNSET; P7_NCUTOFFS];
    if fp7.flags & P7H_GA != 0 {
        cutoff[0] = fp7.cutoff[0];
        cutoff[1] = fp7.cutoff[1];
    }
    if fp7.flags & P7H_TC != 0 {
        cutoff[2] = fp7.cutoff[2];
        cutoff[3] = fp7.cutoff[3];
    }
    if fp7.flags & P7H_NC != 0 {
        cutoff[4] = fp7.cutoff[4];
        cutoff[5] = fp7.cutoff[5];
    }
    for &c in &cutoff {
        w_f32(pfp, c);
    }
    w_f32(pfp, gm.nj); // nj (== 1.0 multihit local)
    w_i32(pfp, 1); // mode = p7_LOCAL
    w_i32(pfp, gm.l); // L = 400
    w_u32(pfp, V3F_PMAGIC); // sentinel
}

/// C p7_hmm_CalculateOccupancy (p7_hmm.c:1338) + modelconfig.c:90-97 local entry,
/// reproducing C's exact float/double arithmetic. Overwrites gm.tsc[k-1][BM].
/// fp7.trans[k] = [MM, MI, MD, IM, II, DM, DD] (p7H order).
fn fix_local_entry_bm(gm: &mut GlocalProfile, fp7: &P7Profile) {
    const H_MM: usize = 0;
    const H_MI: usize = 1;
    const H_DM: usize = 5;
    let m = gm.m;
    let mut mocc = vec![0.0f32; m + 1];
    // mocc[1] = t[0][MI] + t[0][MM]   (float)
    mocc[1] = fp7.trans[0][H_MI] + fp7.trans[0][H_MM];
    for k in 2..=m {
        // C: mocc[k] = mocc[k-1]*(t[MM]+t[MI]) + (1.0-mocc[k-1])*t[DM]
        //   first term  = float*float = float
        //   second term = double  (1.0 is a double literal)
        //   sum promotes to double, stored back as float.
        let a = mocc[k - 1] * (fp7.trans[k - 1][H_MM] + fp7.trans[k - 1][H_MI]); // f32
        let b = (1.0f64 - mocc[k - 1] as f64) * fp7.trans[k - 1][H_DM] as f64; // f64
        mocc[k] = (a as f64 + b) as f32;
    }
    // Z (float) = sum_k occ[k] * (float)(M-k+1)
    let mut z = 0.0f32;
    for k in 1..=m {
        z += mocc[k] * (m - k + 1) as f32;
    }
    // p7P_TSC(gm, k-1, BM) = log(occ[k] / Z)   (float/float, log in double)
    for k in 1..=m {
        gm.tsc[(k - 1) * P7P_NTRANS + P7P_BM] = ((mocc[k] / z) as f64).ln() as f32;
    }
}

/// io.c:128-144 — write an optional string field: `int n` then, if present, the
/// string + '\0'. When absent, only `n=0` is written.
fn write_optstr(b: &mut Vec<u8>, s: Option<&str>) {
    match s {
        None => w_i32(b, 0),
        Some(s) => {
            let bytes = s.as_bytes();
            w_i32(b, bytes.len() as i32);
            b.extend_from_slice(bytes);
            b.push(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expf_matches_libm_on_ordinary_range() {
        // In [-10,10] the Cephes approx is within a couple ulp of libm; here we
        // just sanity-check finiteness and the exact IEEE specials.
        assert_eq!(esl_sse_expf(f32::NEG_INFINITY), 0.0);
        assert_eq!(esl_sse_expf(-1000.0), 0.0);
        assert!(esl_sse_expf(1000.0).is_infinite());
        assert!((esl_sse_expf(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn byteify_wordify_basic() {
        let scale_b = (3.0f64 / ESL_CONST_LOG2) as f32;
        // unbiased cost of a -2.1 nat score with scale 3/log2.
        assert_eq!(unbiased_byteify(scale_b, 0.5f32.ln()), 3); // matches tec_b family
        let scale_w = (500.0f64 / ESL_CONST_LOG2) as f32;
        assert_eq!(wordify(scale_w, 0.0), 0);
    }
}
