// SPDX-License-Identifier: BSD-3-Clause
//! p7_omx — striped-faithful Backward + posterior Decoding + DomainDecoding for
//! the local p7 pipeline that `cmsearch --hmmonly` needs (STEP 2 of the port).
//!
//! Backward is a FAITHFUL simulation of HMMER3's SSE `impl_sse/fwdback.c`
//! `backward_engine` (do_full=TRUE) — NOT a de-striped serial sweep. The striped
//! DD/MD `_mm_move_ss`+shuffle left-shifts and the horizontal xB reductions couple
//! SIMD lanes, and (as established for Forward's DD, `p7_fwdback`) a natural sweep
//! can differ from the striped result by 1 ULP. We therefore replicate the exact
//! striped quad/lane layout (quad q, lane z <-> model node p = q + z*Q + 1) using
//! `[f32;4]` lane arrays that mirror each intrinsic 1:1, then scatter the finished
//! M/I/D cells back to natural [row][k] order for storage/decoding. Per-cell
//! products and the scalar special-state recurrences are layout-independent.
//!
//! Decoding / DomainDecoding are direct transcriptions of `impl_sse/decoding.c`
//! (`p7_Decoding` / `p7_DomainDecoding`), producing the posterior `pp` matrix and
//! the `btot`/`etot`/`mocc` domain-occupancy vectors.
//!
//! Faithfulness anchors:
//!   * C `impl_sse/fwdback.c:467-733` (backward_engine),
//!     `impl_sse/decoding.c:80-210` (p7_Decoding / p7_DomainDecoding),
//!     `impl_sse/p7_oprofile.c:fb_conversion` (striped tfv/rfv packing).
//!   * Cross-checked against the de-striped reference in
//!     `rustyhmmer-dev/src/omx.rs` (backward / decoding / domain_decoding).
//!   * Self-consistency: Forward total == Backward total; Σ posteriors ~ 1.

use crate::p7_fwdback::{nqf, ForwardFilter, Omx, XFactors};

// ---- 4-lane SIMD helpers (bit-identical to the SSE ops they mirror) ----

#[inline(always)]
fn mul4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}
#[inline(always)]
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
#[inline(always)]
fn splat(x: f32) -> [f32; 4] {
    [x, x, x, x]
}
/// Left-shift = `_mm_move_ss(a,0)` then `_mm_shuffle_ps(_,_,_MM_SHUFFLE(0,3,2,1))`:
/// [a0,a1,a2,a3] -> [a1,a2,a3,0]. (Backward's k+1 wrap between striped lanes.)
#[inline(always)]
fn lshift(a: [f32; 4]) -> [f32; 4] {
    [a[1], a[2], a[3], 0.0]
}
/// Horizontal sum reproducing the SSE tree reduction (two shuffle+adds), lane0:
/// (v0+v1)+(v2+v3). Same reduction order as Forward's xE.
#[inline(always)]
fn hsum_tree(v: [f32; 4]) -> f32 {
    // step1: v + shuffle(0,3,2,1)=[v1,v2,v3,v0]
    let w = [v[0] + v[1], v[1] + v[2], v[2] + v[3], v[3] + v[0]];
    // step2: w + shuffle(1,0,3,2)=[w2,w3,w0,w1]; lane0 = w0+w2
    w[0] + w[2]
}

// ---- striped profile (om->tfv, om->rfv) packing, fb_conversion ----

/// Transition-vector index within a quad (p7o_tsc_e). BM..II are the 7 "main"
/// transitions (7 vectors/quad, interleaved); DD is stored in a trailing block.
pub(crate) const T_BM: usize = 0;
pub(crate) const T_MM: usize = 1;
pub(crate) const T_IM: usize = 2;
pub(crate) const T_DM: usize = 3;
pub(crate) const T_MD: usize = 4;
pub(crate) const T_MI: usize = 5;
pub(crate) const T_II: usize = 6;

/// Striped transition table `tfv`, length 8*Q of `[f32;4]` quads. Layout mirrors
/// `om->tfv` exactly: `tfv[q*7 + t]` for t in {BM,MM,IM,DM,MD,MI,II}, then the DD
/// block `tfv[7*Q + q]`. Quad q lane z holds node p = q + z*Q + 1 (0 if p>M).
/// C impl_sse/p7_oprofile.c:fb_conversion.
pub(crate) fn build_tfv(ff: &ForwardFilter, q_n: usize) -> Vec<[f32; 4]> {
    let m = ff.m;
    let mut tfv = vec![[0.0f32; 4]; 8 * q_n];
    for q in 0..q_n {
        let mut bm = [0.0f32; 4];
        let mut mm = [0.0f32; 4];
        let mut im = [0.0f32; 4];
        let mut dm = [0.0f32; 4];
        let mut md = [0.0f32; 4];
        let mut mi = [0.0f32; 4];
        let mut ii = [0.0f32; 4];
        let mut dd = [0.0f32; 4];
        for z in 0..4 {
            let p = q + z * q_n + 1;
            if p <= m {
                // BM/MM/IM/DM (into-node p; valid all p=1..M).
                bm[z] = ff.tbm[p];
                mm[z] = ff.amm[p];
                im[z] = ff.aim[p];
                dm[z] = ff.adm[p];
                // MD/MI/II/DD (out-of-node p; my arrays are already 0 at p=M).
                md[z] = ff.tmd[p];
                mi[z] = ff.tmi[p];
                ii[z] = ff.tii[p];
                dd[z] = ff.tdd[p];
            }
        }
        tfv[q * 7 + T_BM] = bm;
        tfv[q * 7 + T_MM] = mm;
        tfv[q * 7 + T_IM] = im;
        tfv[q * 7 + T_DM] = dm;
        tfv[q * 7 + T_MD] = md;
        tfv[q * 7 + T_MI] = mi;
        tfv[q * 7 + T_II] = ii;
        tfv[7 * q_n + q] = dd;
    }
    tfv
}

/// Striped emission table, `rfv[x*Q + q]` = quad for residue x, quad q; lane z =
/// rfv(node p=q+z*Q+1, x), 0 if p>M. C fb_conversion match-score block.
fn build_rfv_striped(ff: &ForwardFilter, q_n: usize, kp: usize) -> Vec<[f32; 4]> {
    let m = ff.m;
    let mut rfv = vec![[0.0f32; 4]; kp * q_n];
    for x in 0..kp {
        for q in 0..q_n {
            let mut v = [0.0f32; 4];
            for z in 0..4 {
                let p = q + z * q_n + 1;
                if p <= m {
                    v[z] = ff.rfv(p, x);
                }
            }
            rfv[x * q_n + q] = v;
        }
    }
    rfv
}

/// Result of the full-sequence striped Backward: the retained matrix (natural
/// M/I/D + per-row specials/scale) plus the dynamic `has_own_scales` flag and the
/// Backward total score (nats). `bck.xn[0]` is `oxb->xmx[p7X_N]` (used by decoding).
pub struct BackResult {
    pub bck: Omx,
    pub has_own_scales: bool,
    pub bsc: f32,
}

const KP: usize = 18;

/// Full-matrix striped Backward. Faithful simulation of
/// `impl_sse/fwdback.c:467-733 backward_engine(do_full=TRUE)`. `fwd` is the Forward
/// `Omx` (supplies per-row SCALE factors). Dynamic `has_own_scales`: starts FALSE
/// (rows rescaled by the FORWARD scale); if any row's xB exceeds 1e16 the flag
/// flips and lower-i rows use their own `(xB>1e4)?xB:1.0` scale.
pub fn p7_backward(ff: &ForwardFilter, xf: &XFactors, dsq: &[u8], l: usize, fwd: &Omx) -> BackResult {
    let m = ff.m;
    let q_n = nqf(m);
    let tfv = build_tfv(ff, q_n);
    let rfv = build_rfv_striped(ff, q_n, KP);
    let mut bck = Omx::new(m, l);

    let zerov = [0.0f32; 4];
    // Striped rows for the current (dpc) and next (dpp) sequence position.
    let mut mmo_c = vec![zerov; q_n];
    let mut dmo_c = vec![zerov; q_n];
    let mut imo_c = vec![zerov; q_n];
    let mut mmo_p = vec![zerov; q_n]; // prev-row M (read in phase 1)
    let mut imo_p = vec![zerov; q_n]; // prev-row I (read in phase 1)

    // Scatter a striped state row into natural bck[i][p], p = q + z*Q + 1.
    #[inline]
    fn scatter(nat: &mut [f32], src: &[[f32; 4]], q_n: usize, m: usize) {
        for q in 0..q_n {
            for z in 0..4 {
                let p = q + z * q_n + 1;
                if p <= m {
                    nat[p] = src[q][z];
                }
            }
        }
        nat[0] = 0.0;
    }

    let mut has_own_scales = false;
    let mut totscale;

    // ---- Initialize the L row (fwdback.c:486-560). ----
    let mut xj = 0.0f32;
    let mut xb = 0.0f32;
    let mut xn = 0.0f32;
    let mut xc = xf.c_move; // C<-T
    let mut xe = xc * xf.e_move; // E<-C, no tail
    let xev = splat(xe);
    for q in 0..q_n {
        mmo_c[q] = xev;
        dmo_c[q] = xev;
        imo_c[q] = zerov;
    }
    // DD paths (fwdback.c:512-533): pass 1 includes xE (from DMO(q)); passes 2..4
    // extend the DD component only, carrying dcv. tp walks DD(Q-1)..DD(0).
    {
        // pass 1
        let mut dpv = lshift(dmo_c[q_n - 1]);
        let mut dcv = zerov;
        for q in (0..q_n).rev() {
            dcv = mul4(dpv, tfv[7 * q_n + q]); // *DD(q)
            dmo_c[q] = add4(dmo_c[q], dcv);
            dpv = dmo_c[q];
        }
        // passes 2..4 (carried dcv, left-shifted each pass)
        for _j in 1..4 {
            dcv = lshift(dcv);
            for q in (0..q_n).rev() {
                dcv = mul4(dcv, tfv[7 * q_n + q]);
                dmo_c[q] = add4(dmo_c[q], dcv);
            }
        }
    }
    // MD init: MMO(q) += dcv * MD(q), dcv walks DMO. tp = MD(Q-1)=7Q-3, tp-=7.
    {
        let mut dcvm = lshift(dmo_c[0]);
        for q in (0..q_n).rev() {
            mmo_c[q] = add4(mmo_c[q], mul4(dcvm, tfv[q * 7 + T_MD]));
            dcvm = dmo_c[q];
        }
    }
    // Sparse rescale L row by fwd scale (has_own_scales=FALSE at L).
    let scl = fwd.scale[l];
    if scl > 1.0 {
        let inv = 1.0 / scl;
        xe *= inv;
        xn *= inv;
        xc *= inv;
        xj *= inv;
        xb *= inv;
        let invv = splat(inv);
        for q in 0..q_n {
            mmo_c[q] = mul4(mmo_c[q], invv);
            dmo_c[q] = mul4(dmo_c[q], invv);
            imo_c[q] = mul4(imo_c[q], invv);
        }
    }
    bck.scale[l] = scl;
    totscale = (scl as f64).ln() as f32;
    bck.xe[l] = xe;
    bck.xn[l] = xn;
    bck.xj[l] = xj;
    bck.xb[l] = xb;
    bck.xc[l] = xc;
    scatter(&mut bck.mmx[l], &mmo_c, q_n, m);
    scatter(&mut bck.dmx[l], &dmo_c, q_n, m);
    scatter(&mut bck.imx[l], &imo_c, q_n, m);

    // Roll L row into "previous".
    std::mem::swap(&mut mmo_p, &mut mmo_c);
    std::mem::swap(&mut imo_p, &mut imo_c);

    // ---- Main recursion i = L-1 .. 1 (fwdback.c:562-693). ----
    for i in (1..l).rev() {
        let xi1 = dsq[i + 1] as usize; // residue x_{i+1}
        let rp_base = xi1 * q_n;

        // phase 1: collect B(i), build I(i,k) and partial {M,D}(i,k).
        // Left-shifted first transition quads and mpv (M(i+1,quad0)*e).
        let mut tmmv = lshift(tfv[0 * 7 + T_MM]); // MM(0)
        let mut timv = lshift(tfv[0 * 7 + T_IM]); // IM(0)
        let mut tdmv = lshift(tfv[0 * 7 + T_DM]); // DM(0)
        let mut mpv = lshift(mul4(mmo_p[0], rfv[rp_base + 0]));
        let mut xbv = zerov;
        for q in (0..q_n).rev() {
            let ipv = imo_p[q];
            // I(i,q) = I(i+1,q)*II(q) + mpv*IM(k+1)
            imo_c[q] = add4(mul4(ipv, tfv[q * 7 + T_II]), mul4(mpv, timv));
            // partial D(i,q) = mpv*DM(k+1)
            dmo_c[q] = mul4(mpv, tdmv);
            // partial M(i,q) = I(i+1,q)*MI(q) + mpv*MM(k+1)
            let mcv = add4(mul4(ipv, tfv[q * 7 + T_MI]), mul4(mpv, tmmv));
            // mpv := M(i+1,q)*e(quad q)  (for next-lower q)
            mpv = mul4(mmo_p[q], rfv[rp_base + q]);
            mmo_c[q] = mcv;
            // reload transition quads for next-lower q.
            tdmv = tfv[q * 7 + T_DM];
            timv = tfv[q * 7 + T_IM];
            tmmv = tfv[q * 7 + T_MM];
            // xBv += mpv * BM(q)
            xbv = add4(xbv, mul4(mpv, tfv[q * 7 + T_BM]));
        }

        // phase 2: specials from xB (horizontal sum), then N/J/C/E.
        xb = hsum_tree(xbv);
        xc = xc * xf.c_loop;
        xj = (xb * xf.j_move) + (xj * xf.j_loop);
        xn = (xb * xf.n_move) + (xn * xf.n_loop);
        xe = (xc * xf.e_move) + (xj * xf.e_loop);
        let xev = splat(xe);

        // phase 3: {M,D}->E paths + one DD step; add xE into M.
        {
            let mut dpv = add4(dmo_c[0], xev);
            dpv = lshift(dpv);
            let mut dcv3 = zerov;
            for q in (0..q_n).rev() {
                dcv3 = mul4(dpv, tfv[7 * q_n + q]); // DD(q)
                dmo_c[q] = add4(dmo_c[q], add4(dcv3, xev));
                dpv = dmo_c[q];
                mmo_c[q] = add4(mmo_c[q], xev);
            }
            // phase 4: finish DD (3 more passes), carried dcv.
            for _j in 1..4 {
                dcv3 = lshift(dcv3);
                for q in (0..q_n).rev() {
                    dcv3 = mul4(dcv3, tfv[7 * q_n + q]);
                    dmo_c[q] = add4(dmo_c[q], dcv3);
                }
            }
        }
        // phase 5: add M->D paths.
        {
            let mut dcv5 = lshift(dmo_c[0]);
            for q in (0..q_n).rev() {
                mmo_c[q] = add4(mmo_c[q], mul4(dcv5, tfv[q * 7 + T_MD]));
                dcv5 = dmo_c[q];
            }
        }

        // Sparse rescale (fwdback.c:651-678), dynamic has_own_scales.
        if xb > 1.0e16 {
            has_own_scales = true;
        }
        let sc_i = if has_own_scales {
            if xb > 1.0e4 {
                xb
            } else {
                1.0
            }
        } else {
            fwd.scale[i]
        };
        if sc_i > 1.0 {
            let inv = 1.0 / sc_i;
            xe *= inv;
            xn *= inv;
            xj *= inv;
            xb *= inv;
            xc *= inv;
            let invv = splat(inv);
            for q in 0..q_n {
                mmo_c[q] = mul4(mmo_c[q], invv);
                dmo_c[q] = mul4(dmo_c[q], invv);
                imo_c[q] = mul4(imo_c[q], invv);
            }
            totscale += (sc_i as f64).ln() as f32;
        }
        bck.scale[i] = sc_i;
        bck.xe[i] = xe;
        bck.xn[i] = xn;
        bck.xj[i] = xj;
        bck.xb[i] = xb;
        bck.xc[i] = xc;
        scatter(&mut bck.mmx[i], &mmo_c, q_n, m);
        scatter(&mut bck.dmx[i], &dmo_c, q_n, m);
        scatter(&mut bck.imx[i], &imo_c, q_n, m);

        // Roll current into previous.
        std::mem::swap(&mut mmo_p, &mut mmo_c);
        std::mem::swap(&mut imo_p, &mut imo_c);
    }

    // ---- Termination at i=0 (only N,B reachable) (fwdback.c:695-733). ----
    let x1 = dsq[1] as usize;
    let rp0 = x1 * q_n;
    let mut xbv = zerov;
    for q in 0..q_n {
        let mut mpv = mul4(mmo_p[q], rfv[rp0 + q]);
        mpv = mul4(mpv, tfv[q * 7 + T_BM]); // BM(q), tp += 7 stride
        xbv = add4(xbv, mpv);
    }
    xb = hsum_tree(xbv);
    xn = (xb * xf.n_move) + (xn * xf.n_loop);
    bck.xb[0] = xb;
    bck.xc[0] = 0.0;
    bck.xj[0] = 0.0;
    bck.xn[0] = xn;
    bck.xe[0] = 0.0;
    bck.scale[0] = 1.0;
    // Row 0 M/I/D are zero (already from Omx::new).
    bck.totscale = totscale;

    let bsc = totscale + (xn as f64).ln() as f32;
    BackResult {
        bck,
        has_own_scales,
        bsc,
    }
}

/// Posterior decoding matrix (pp). C `impl_sse/decoding.c:p7_Decoding`. M/I cells
/// = fwd*bck*totrv (D cells 0); specials N/J/C. `scaleproduct` starts 1/bck.xn[0]
/// and updates per row when `has_own_scales`. Returns the pp `Omx` (M/I in mmx/imx,
/// specials in xn/xj/xc). Panics never; overflow → pp may contain NaN (mirrors C's
/// eslERANGE, which the caller treats as "ignore this region").
pub fn p7_decoding(
    fwd: &Omx,
    bck: &Omx,
    xf: &XFactors,
    has_own_scales: bool,
) -> Omx {
    let m = fwd.m;
    let l = fwd.ld;
    let mut pp = Omx::new(m, l);
    let mut scaleproduct = 1.0f32 / bck.xn[0];
    // Row 0 already zero.
    for i in 1..=l {
        let totrv = scaleproduct * fwd.scale[i];
        for k in 1..=m {
            pp.mmx[i][k] = fwd.mmx[i][k] * bck.mmx[i][k] * totrv;
            pp.imx[i][k] = fwd.imx[i][k] * bck.imx[i][k] * totrv;
            // D cells stay 0.
        }
        pp.xe[i] = 0.0;
        pp.xn[i] = fwd.xn[i - 1] * bck.xn[i] * xf.n_loop * scaleproduct;
        pp.xj[i] = fwd.xj[i - 1] * bck.xj[i] * xf.j_loop * scaleproduct;
        pp.xc[i] = fwd.xc[i - 1] * bck.xc[i] * xf.c_loop * scaleproduct;
        pp.xb[i] = 0.0;
        if has_own_scales {
            scaleproduct *= fwd.scale[i] / bck.scale[i];
        }
    }
    pp
}

/// Domain-occupancy decoding. C `impl_sse/decoding.c:p7_DomainDecoding`. Fills
/// `btot[i]` (cumulative begin), `etot[i]` (cumulative end), `mocc[i]` (per-residue
/// prob of being emitted by the core model = 1 - N/J/C occupancy). Uses only the
/// special states of `fwd`/`bck` plus scale factors.
pub struct DomainDecode {
    pub l: usize,
    pub btot: Vec<f32>, // 0..=L
    pub etot: Vec<f32>,
    pub mocc: Vec<f32>,
}

pub fn p7_domain_decoding(
    fwd: &Omx,
    bck: &Omx,
    xf: &XFactors,
    has_own_scales: bool,
) -> DomainDecode {
    let l = fwd.ld;
    let mut btot = vec![0.0f32; l + 1];
    let mut etot = vec![0.0f32; l + 1];
    let mut mocc = vec![0.0f32; l + 1];
    let mut scaleproduct = 1.0f32 / bck.xn[0];
    for i in 1..=l {
        // scaleproduct is prod_{j=0}^{i-2} here.
        btot[i] = btot[i - 1]
            + (fwd.xb[i - 1] * bck.xb[i - 1] * fwd.scale[i - 1] * scaleproduct);
        if has_own_scales {
            scaleproduct *= fwd.scale[i - 1] / bck.scale[i - 1];
        }
        // scaleproduct is prod_{j=0}^{i-1} now.
        etot[i] = etot[i - 1]
            + (fwd.xe[i] * bck.xe[i] * fwd.scale[i] * scaleproduct);
        let mut njcp = fwd.xn[i - 1] * bck.xn[i] * xf.n_loop * scaleproduct;
        njcp += fwd.xj[i - 1] * bck.xj[i] * xf.j_loop * scaleproduct;
        njcp += fwd.xc[i - 1] * bck.xc[i] * xf.c_loop * scaleproduct;
        mocc[i] = 1.0 - njcp;
    }
    DomainDecode { l, btot, etot, mocc }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p7_fwdback::{build_forward_filter, forward_score, p7_forward};
    use crate::p7_hmm::P7Profile;

    fn synthetic_p7(m: usize, seed: u64) -> P7Profile {
        let mut s = seed;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut p7 = P7Profile::new(m as i32);
        for k in 1..=m {
            let mut e = [0.0f32; 4];
            let mut sum = 0.0f32;
            for x in 0..4 {
                e[x] = (0.05 + rng()) as f32;
                sum += e[x];
            }
            for x in 0..4 {
                e[x] /= sum;
            }
            p7.mat[k] = e;
        }
        for k in 0..=m {
            if k == m {
                p7.trans[k] = [1.0, 0.0, 0.0, 0.75, 0.25, 0.72, 0.28];
            } else {
                p7.trans[k] = [0.80, 0.12, 0.08, 0.75, 0.25, 0.72, 0.28];
            }
        }
        p7
    }

    fn rand_dsq(l: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut v = vec![0u8; l + 2];
        v[0] = 4;
        v[l + 1] = 4;
        for i in 1..=l {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v[i] = (s % 4) as u8;
        }
        v
    }

    // Self-consistency: Forward total score == Backward total score. Both compute
    // log P(seq | model); they use different accumulation paths so we allow a tiny
    // relative tolerance (should be within a few ULP-scaled units).
    #[test]
    fn forward_total_equals_backward_total() {
        for &m in &[7usize, 20, 33, 64, 101, 128, 200] {
            for seed in 0..3u64 {
                let p7 = synthetic_p7(m, 0x51 ^ (m as u64) << 3 ^ seed);
                let ff = build_forward_filter(&p7);
                for &l in &[10usize, 50, 200] {
                    let dsq = rand_dsq(l, 0xC0DE ^ (m as u64) << 8 ^ (l as u64) << 2 ^ seed);
                    let xf = XFactors::multihit(l);
                    let fx = p7_forward(&ff, &xf, &dsq, l);
                    let fsc = forward_score(&fx, &xf);
                    let br = p7_backward(&ff, &xf, &dsq, l, &fx);
                    let bsc = br.bsc;
                    let diff = (fsc - bsc).abs();
                    let tol = 1e-3 * fsc.abs().max(1.0);
                    assert!(
                        diff <= tol,
                        "fwd/bck total mismatch m={m} l={l} seed={seed}: fsc={fsc} bsc={bsc} diff={diff}"
                    );
                }
            }
        }
    }

    // Posterior occupancy is a probability: mocc[i] in [0,1] (within fp slack), and
    // the residue is emitted by either the core model (mocc) or N/J/C (1-mocc).
    #[test]
    fn mocc_is_a_probability() {
        let m = 64usize;
        let p7 = synthetic_p7(m, 0xABCDEF);
        let ff = build_forward_filter(&p7);
        let l = 120usize;
        let dsq = rand_dsq(l, 0x7777);
        let xf = XFactors::multihit(l);
        let fx = p7_forward(&ff, &xf, &dsq, l);
        let br = p7_backward(&ff, &xf, &dsq, l, &fx);
        let dd = p7_domain_decoding(&fx, &br.bck, &xf, br.has_own_scales);
        for i in 1..=l {
            assert!(
                dd.mocc[i] >= -1e-3 && dd.mocc[i] <= 1.0 + 1e-3,
                "mocc[{i}] out of range: {}",
                dd.mocc[i]
            );
        }
        // btot/etot are nondecreasing cumulative sums.
        for i in 1..=l {
            assert!(dd.btot[i] >= dd.btot[i - 1] - 1e-4, "btot not monotone at {i}");
            assert!(dd.etot[i] >= dd.etot[i - 1] - 1e-4, "etot not monotone at {i}");
        }
    }

    // Per-position posterior: Σ_k pp_M(i,k) + Σ_k pp_I(i,k) + (N/J/C occupancy) ≈ 1.
    #[test]
    fn posteriors_sum_to_one() {
        let m = 40usize;
        let p7 = synthetic_p7(m, 0x1111_2222);
        let ff = build_forward_filter(&p7);
        let l = 90usize;
        let dsq = rand_dsq(l, 0x3333);
        let xf = XFactors::multihit(l);
        let fx = p7_forward(&ff, &xf, &dsq, l);
        let br = p7_backward(&ff, &xf, &dsq, l, &fx);
        let pp = p7_decoding(&fx, &br.bck, &xf, br.has_own_scales);
        for i in 1..=l {
            let mut s = pp.xn[i] + pp.xj[i] + pp.xc[i];
            for k in 1..=m {
                s += pp.mmx[i][k] + pp.imx[i][k];
            }
            assert!(
                (s - 1.0).abs() < 5e-3,
                "posterior row {i} sums to {s}, expected ~1"
            );
        }
    }

    // Independent DE-STRIPED backward reference, transcribed verbatim from
    // rustyhmmer-dev/src/omx.rs::backward_full (a natural serial sweep + natural xB
    // sum). Used ONLY to cross-check my striped backward: they should agree to a
    // tight relative tolerance (they differ at most ~1 ULP because the de-striped
    // DD/xE grouping and xB summation order diverge from the striped lanes — which
    // is exactly why the shipped engine is striped-faithful for strict C parity).
    fn ref_backward_destriped(
        ff: &ForwardFilter,
        xf: &XFactors,
        dsq: &[u8],
        ld: usize,
        fwd: &Omx,
    ) -> (Omx, bool) {
        let m = ff.m;
        let mut bck = Omx::new(m, ld);
        let mut has_own_scales = false;
        let mut xj = 0.0f32;
        let mut xb = 0.0f32;
        let mut xn = 0.0f32;
        let mut xc = xf.c_move;
        let mut xe = xc * xf.e_move;
        {
            let mc = &mut bck.mmx[ld];
            let dc = &mut bck.dmx[ld];
            mc[m] = xe;
            dc[m] = xe;
            for k in (1..m).rev() {
                dc[k] = xe + dc[k + 1] * ff.tdd[k];
                mc[k] = xe + dc[k + 1] * ff.tmd[k];
            }
        }
        let scl = fwd.scale[ld];
        if scl > 1.0 {
            let inv = 1.0 / scl;
            xe *= inv;
            xn *= inv;
            xj *= inv;
            xb *= inv;
            xc *= inv;
            for k in 0..=m {
                bck.mmx[ld][k] *= inv;
                bck.dmx[ld][k] *= inv;
                bck.imx[ld][k] *= inv;
            }
        }
        bck.scale[ld] = scl;
        bck.xe[ld] = xe;
        bck.xn[ld] = xn;
        bck.xj[ld] = xj;
        bck.xb[ld] = xb;
        bck.xc[ld] = xc;
        for i in (1..ld).rev() {
            let x = dsq[i + 1] as usize;
            let mut b = 0.0f32;
            for k in 1..=m {
                b += bck.mmx[i + 1][k] * ff.tbm[k] * ff.rfv(k, x);
            }
            xb = b;
            xc *= xf.c_loop;
            xj = (xb * xf.j_move) + (xj * xf.j_loop);
            xn = (xb * xf.n_move) + (xn * xf.n_loop);
            xe = (xc * xf.e_move) + (xj * xf.e_loop);
            {
                let mm_next = bck.mmx[i + 1].clone();
                let im_next = bck.imx[i + 1].clone();
                let mut mc = vec![0.0f32; m + 1];
                let mut dc = vec![0.0f32; m + 1];
                let mut ic = vec![0.0f32; m + 1];
                mc[m] = xe;
                dc[m] = xe;
                for k in (1..m).rev() {
                    let m_emit = mm_next[k + 1] * ff.rfv(k + 1, x);
                    mc[k] = m_emit * ff.amm[k + 1] + im_next[k] * ff.tmi[k] + xe + dc[k + 1] * ff.tmd[k];
                    ic[k] = m_emit * ff.aim[k + 1] + im_next[k] * ff.tii[k];
                    dc[k] = m_emit * ff.adm[k + 1] + dc[k + 1] * ff.tdd[k] + xe;
                }
                bck.mmx[i] = mc;
                bck.dmx[i] = dc;
                bck.imx[i] = ic;
            }
            if xb > 1.0e16 {
                has_own_scales = true;
            }
            let scl = if has_own_scales {
                if xb > 1.0e4 {
                    xb
                } else {
                    1.0
                }
            } else {
                fwd.scale[i]
            };
            if scl > 1.0 {
                let inv = 1.0 / scl;
                xe *= inv;
                xn *= inv;
                xj *= inv;
                xb *= inv;
                xc *= inv;
                for k in 0..=m {
                    bck.mmx[i][k] *= inv;
                    bck.dmx[i][k] *= inv;
                    bck.imx[i][k] *= inv;
                }
            }
            bck.scale[i] = scl;
            bck.xe[i] = xe;
            bck.xn[i] = xn;
            bck.xj[i] = xj;
            bck.xb[i] = xb;
            bck.xc[i] = xc;
        }
        let x1 = dsq[1] as usize;
        let mut b0 = 0.0f32;
        for k in 1..=m {
            b0 += bck.mmx[1][k] * ff.rfv(k, x1) * ff.tbm[k];
        }
        xn = (b0 * xf.n_move) + (xn * xf.n_loop);
        bck.xb[0] = b0;
        bck.xn[0] = xn;
        bck.scale[0] = 1.0;
        (bck, has_own_scales)
    }

    // Cross-check my striped backward + domain decoding against the independent
    // de-striped reference on btot/etot/mocc and the special-state rows.
    #[test]
    fn striped_backward_matches_destriped_reference() {
        for &m in &[7usize, 20, 33, 64, 101, 128, 200] {
            for seed in 0..3u64 {
                let p7 = synthetic_p7(m, 0x9E ^ (m as u64) << 3 ^ seed);
                let ff = build_forward_filter(&p7);
                for &l in &[10usize, 60, 200] {
                    let dsq = rand_dsq(l, 0x2468 ^ (m as u64) << 8 ^ (l as u64) << 2 ^ seed);
                    let xf = XFactors::multihit(l);
                    let fx = p7_forward(&ff, &xf, &dsq, l);
                    let br = p7_backward(&ff, &xf, &dsq, l, &fx);
                    let (rb, rown) = ref_backward_destriped(&ff, &xf, &dsq, l, &fx);
                    assert_eq!(br.has_own_scales, rown, "own_scales m={m} l={l}");
                    let dd = p7_domain_decoding(&fx, &br.bck, &xf, br.has_own_scales);
                    let rdd = p7_domain_decoding(&fx, &rb, &xf, rown);
                    // Compare btot/etot/mocc to a tight relative tolerance.
                    let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
                    for i in 0..=l {
                        assert!(rel(dd.btot[i], rdd.btot[i]) < 2e-4, "btot m={m} l={l} i={i}: {} vs {}", dd.btot[i], rdd.btot[i]);
                        assert!(rel(dd.etot[i], rdd.etot[i]) < 2e-4, "etot m={m} l={l} i={i}: {} vs {}", dd.etot[i], rdd.etot[i]);
                        assert!((dd.mocc[i] - rdd.mocc[i]).abs() < 2e-4, "mocc m={m} l={l} i={i}: {} vs {}", dd.mocc[i], rdd.mocc[i]);
                    }
                    // Special-state backward N total (feeds decoding normalization).
                    assert!(rel(br.bck.xn[0], rb.xn[0]) < 2e-4, "xN0 m={m} l={l}");
                }
            }
        }
    }

    // Real-model cross-check (gated on INFERNOX_TEST_CM): Forward total == Backward
    // total on an actual filter HMM.
    #[test]
    fn real_cm_fwd_bck_consistency() {
        let path = match std::env::var("INFERNOX_TEST_CM") {
            Ok(p) => p,
            Err(_) => return,
        };
        let cm = crate::cm_file::cm_file_read(&path).expect("read cm");
        let p7 = cm.p7.as_ref().expect("p7 filter");
        let ff = build_forward_filter(p7);
        for &l in &[40usize, 120, 300] {
            let dsq = rand_dsq(l, 0xBEEF ^ l as u64);
            let xf = XFactors::multihit(l);
            let fx = p7_forward(&ff, &xf, &dsq, l);
            let fsc = forward_score(&fx, &xf);
            let br = p7_backward(&ff, &xf, &dsq, l, &fx);
            let diff = (fsc - br.bsc).abs();
            let tol = 1e-3 * fsc.abs().max(1.0);
            assert!(
                diff <= tol,
                "real-cm fwd/bck mismatch m={} l={l}: fsc={fsc} bsc={}",
                p7.m,
                br.bsc
            );
        }
        eprintln!("real-cm fwd/bck OK: m={}", p7.m);
    }
}
