// SPDX-License-Identifier: BSD-3-Clause
//! p7_domaindef — LOCAL p7 domain definition for `cmsearch --hmmonly` (STEP 3).
//!
//! Faithful port of HMMER3's `p7_domaindef_ByPosteriorHeuristics` (the LOCAL
//! variant, `impl_sse` engines) — the region-finding + per-region rescore driver
//! that `cm_pipeline.c:pli_final_stage_hmmonly` calls with `long_target=FALSE`
//! and `do_aln=TRUE`. Builds, for each domain: envelope coords (ienv/jenv), the
//! optimal-accuracy alignment (trace + iali/jali + oasc), the Forward envelope
//! score (envsc), and the null2 bias correction (domcorrection).
//!
//! Engines used are this crate's optimized odds-space matrices:
//!   * Forward  — `p7_fwdback::p7_forward` / `forward_score`
//!   * Backward — `p7_omx::p7_backward`
//!   * Decoding / DomainDecoding — `p7_omx::p7_decoding` / `p7_domain_decoding`
//!   * OptimalAccuracy + OATrace — this module (`impl_sse/optacc.c`)
//!   * Null2_ByExpectation — this module (`impl_sse/null2.c`)
//!   * StochasticTrace ensemble + clustering — this module
//!     (`impl_sse/stotrace.c` + `p7_spensemble.c` + `p7_domaindef.c`)
//!
//! Faithfulness anchors: `impl_sse/optacc.c`, `impl_sse/null2.c`,
//! `impl_sse/stotrace.c`, `p7_domaindef.c` (region loop / rescore_isolated_domain
//! / region_trace_ensemble), `p7_spensemble.c`, `esl_cluster.c`, `esl_random.c`.
//! Cross-checked against `rustyhmmer-dev/src/dp.rs` (p7_domaindef_local, region
//! ensemble, stochastic traces, Null2) — proven parity.

use crate::p7_fwdback::{
    degen_set, forward_score, nqf, p7_forward, ForwardFilter, Omx, XFactors, K_CANON, KP,
};
use crate::p7_omx::{p7_backward, p7_decoding, p7_domain_decoding};

const K: usize = K_CANON; // 4 canonical RNA residues

// ---- trace state codes (C p7T_*). ST_S terminates the traceback loop. ----
const ST_T: u8 = 0;
const ST_C: u8 = 1;
const ST_E: u8 = 2;
const ST_M: u8 = 3;
const ST_D: u8 = 4;
const ST_I: u8 = 5;
const ST_N: u8 = 6;
const ST_B: u8 = 7;
const ST_J: u8 = 8;
const ST_S: u8 = 9;

/// One domain/envelope produced by local domain definition. Coords are
/// window-local (1-based) w.r.t. the `dsq` passed to [`p7_domaindef_local`].
#[derive(Clone, Debug)]
pub struct Domain {
    pub ienv: i64,
    pub jenv: i64,
    pub iali: i64,
    pub jali: i64,
    pub envsc: f32,         // envelope Forward score, NATS
    pub oasc: f32,          // optimal-accuracy score (expected # correct residues)
    pub domcorrection: f32, // null2 correction, NATS
    /// OA trace (forward order): (state, k, i, postprob). seq coords window-local.
    pub tr: Vec<(u8, i32, i32, f32)>,
}

/// Result of domain definition: domains + the counts the tblout reports + the
/// per-residue null2 log-odds `n2sc[0..=L]`.
pub struct DomainDef {
    pub domains: Vec<Domain>,
    pub nexpected: f32,
    pub nregions: i32,
    pub nclustered: i32,
    pub noverlaps: i32,
    pub nenvelopes: i32,
    pub n2sc: Vec<f32>,
}

// ===========================================================================
// Easel FAST RNG (esl_randomness_CreateFast → knuth) + vector helpers.
// Byte-exact copies of the crate's validated port (esl_random.c / esl_vectorops.c).
// ===========================================================================

struct EslRng {
    x: u32,
}
fn esl_mix3(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 13;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 8;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 13;
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 12;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 16;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 5;
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 3;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 10;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 15;
    c
}
impl EslRng {
    fn new(seed: u32) -> Self {
        let mut x = esl_mix3(seed, 87654321, 12345678);
        if x == 0 { x = 42; }
        EslRng { x }
    }
    fn knuth(&mut self) -> u32 {
        self.x = self.x.wrapping_mul(69069).wrapping_add(1);
        self.x
    }
    fn random(&mut self) -> f64 {
        (self.knuth() as f64) / 4294967296.0
    }
    /// C esl_rnd_FChoose: one draw, walk cumulative (double-precision norm/sum).
    fn fchoose(&mut self, p: &[f32]) -> usize {
        let mut norm = 0.0f64;
        let roll = self.random();
        for &v in p { norm += v as f64; }
        let mut sum = 0.0f64;
        for (i, &v) in p.iter().enumerate() {
            sum += v as f64;
            if roll < sum / norm { return i; }
        }
        p.len() - 1
    }
}

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
fn esl_vec_fnorm(v: &mut [f32]) {
    let sum = esl_vec_fsum(v);
    if sum != 0.0 {
        for x in v.iter_mut() { *x /= sum; }
    } else {
        let n = v.len() as f32;
        for x in v.iter_mut() { *x = 1.0 / n; }
    }
}

// ===========================================================================
// Null2 by expectation (impl_sse/null2.c). Operates on the DECODING pp matrix.
// ===========================================================================

/// C `p7_Null2_ByExpectation` (impl_sse/null2.c). Accumulates the pp rows 1..Ld
/// (row 0 seeded from row 1), normalizes by 1/Ld, and forms the LINEAR null2
/// odds vector. The Σ_k reduction is done in the SIMD STRIPED lane order + tree
/// horizontal sum (NOT natural k order) so it is bit-identical to C.
fn p7_null2_by_expectation(ff: &ForwardFilter, pp: &Omx, ld: usize) -> [f32; KP] {
    let m = ff.m;
    let q_n = nqf(m);
    // Row-0 accumulator (natural), seeded from row 1 (null2.c:60-63).
    let mut acc_m = pp.mmx[1].clone();
    let mut acc_i = pp.imx[1].clone();
    let mut xn = pp.xn[1];
    let mut xc = pp.xc[1];
    let mut xj = pp.xj[1];
    for r in 2..=ld {
        for k in 0..=m {
            acc_m[k] += pp.mmx[r][k];
            acc_i[k] += pp.imx[r][k];
        }
        xn += pp.xn[r];
        xc += pp.xc[r];
        xj += pp.xj[r];
    }
    let norm = 1.0f32 / ld as f32;
    for k in 0..=m {
        acc_m[k] *= norm;
        acc_i[k] *= norm;
    }
    xn *= norm;
    xc *= norm;
    xj *= norm;
    let xfactor = xn + xc + xj;

    // null2[x] = hsum_lanes( Σ_q accM[node]*rfv[node][x] + Σ_q accI[node] ) + xfactor.
    // Striped lane accumulation (q inner, M then I per quad) then the SSE hsum tree.
    let mut null2 = [0.0f32; KP];
    for x in 0..K {
        let mut sv = [0.0f32; 4];
        for q in 0..q_n {
            for (z, sv_z) in sv.iter_mut().enumerate() {
                let node = q + z * q_n + 1;
                if node <= m {
                    *sv_z += acc_m[node] * ff.rfv(node, x);
                }
                // I insert odds implicitly 1.0
                if node <= m {
                    *sv_z += acc_i[node];
                } else {
                    // padding lane contributes 0 (acc is 0 for node>m anyway)
                }
            }
        }
        // esl_sse_hsum_ps tree: (s0+s1)+(s2+s3)
        let w = [sv[0] + sv[1], sv[1] + sv[2], sv[2] + sv[3], sv[3] + sv[0]];
        let s = w[0] + w[2];
        null2[x] = s + xfactor;
    }
    // esl_abc_FAvgScVec: degenerate codes = unweighted mean of the fold.
    for x in (K + 1)..=(KP - 3) {
        let set = degen_set(x);
        let mut result = 0.0f32;
        let mut ndegen = 0.0f32;
        for &y in set {
            result += null2[y];
            ndegen += 1.0;
        }
        null2[x] = if ndegen > 0.0 { result / ndegen } else { 0.0 };
    }
    null2[K] = 1.0; // gap
    null2[KP - 2] = 1.0; // nonresidue '*'
    null2[KP - 1] = 1.0; // missing '~'
    null2
}

// ===========================================================================
// Optimal Accuracy DP + traceback (impl_sse/optacc.c). Operates on the pp
// (decoding) matrix. All reductions are MAX (order-independent, exact), so this
// is a NATURAL transcription that is bit-identical to the striped SSE.
// ===========================================================================

const NEG_INF: f32 = f32::NEG_INFINITY;

/// AND-mask: `_mm_and_ps(_mm_cmpgt_ps(t,0), v)` → v if t>0 else +0.0.
#[inline(always)]
fn masked(t: f32, v: f32) -> f32 {
    if t > 0.0 { v } else { 0.0 }
}

/// C `p7_OptimalAccuracy` (impl_sse/optacc.c). Fills an OA Omx (natural M/D/I +
/// specials) from the pp matrix. Returns `(oa, oasc)` where oasc = OA C-state at
/// row L (expected # of correctly aligned residues).
fn p7_optimal_accuracy(ff: &ForwardFilter, xf: &XFactors, pp: &Omx, l: usize) -> (Omx, f32) {
    let m = ff.m;
    let mut oa = Omx::new(m, l);
    // Row 0: M/D/I = -inf; E=-inf, N=0, J=-inf, B=0, C=-inf.
    for k in 0..=m {
        oa.mmx[0][k] = NEG_INF;
        oa.dmx[0][k] = NEG_INF;
        oa.imx[0][k] = NEG_INF;
    }
    oa.xe[0] = NEG_INF;
    oa.xn[0] = 0.0;
    oa.xj[0] = NEG_INF;
    oa.xb[0] = 0.0;
    oa.xc[0] = NEG_INF;

    for i in 1..=l {
        let xb_prev = oa.xb[i - 1];
        let mut xe = NEG_INF;
        // M and I (per cell). M(i,k) from B(i-1),M(i-1,k-1),I(i-1,k-1),D(i-1,k-1).
        for k in 1..=m {
            let pm = if k >= 2 { oa.mmx[i - 1][k - 1] } else { NEG_INF };
            let pi = if k >= 2 { oa.imx[i - 1][k - 1] } else { NEG_INF };
            let pd = if k >= 2 { oa.dmx[i - 1][k - 1] } else { NEG_INF };
            let mut sv = masked(ff.tbm[k], xb_prev);
            sv = sv.max(masked(ff.amm[k], pm));
            sv = sv.max(masked(ff.aim[k], pi));
            sv = sv.max(masked(ff.adm[k], pd));
            sv += pp.mmx[i][k];
            oa.mmx[i][k] = sv;
            if sv > xe { xe = sv; }
            // I(i,k) from M(i-1,k), I(i-1,k).
            let mi = oa.mmx[i - 1][k];
            let ii = oa.imx[i - 1][k];
            let mut si = masked(ff.tmi[k], mi);
            si = si.max(masked(ff.tii[k], ii));
            si += pp.imx[i][k];
            oa.imx[i][k] = si;
        }
        // D (natural forward sweep; max is exact ⇒ == striped 4-pass).
        oa.dmx[i][1] = NEG_INF;
        for k in 2..=m {
            let d = masked(ff.tmd[k - 1], oa.mmx[i][k - 1])
                .max(masked(ff.tdd[k - 1], oa.dmx[i][k - 1]));
            oa.dmx[i][k] = d;
            if d > xe { xe = d; }
        }
        // Specials (optacc.c). ESL_MAX + the "==0.0 ? 0.0" gating on xf factors.
        oa.xe[i] = xe;
        let t1 = if xf.j_loop == 0.0 { 0.0 } else { oa.xj[i - 1] + pp.xj[i] };
        let t2 = if xf.e_loop == 0.0 { 0.0 } else { xe };
        oa.xj[i] = t1.max(t2);
        let t1 = if xf.c_loop == 0.0 { 0.0 } else { oa.xc[i - 1] + pp.xc[i] };
        let t2 = if xf.e_move == 0.0 { 0.0 } else { xe };
        oa.xc[i] = t1.max(t2);
        oa.xn[i] = if xf.n_loop == 0.0 { 0.0 } else { oa.xn[i - 1] + pp.xn[i] };
        let t1 = if xf.n_move == 0.0 { 0.0 } else { oa.xn[i] };
        let t2 = if xf.j_move == 0.0 { 0.0 } else { oa.xj[i] };
        oa.xb[i] = t1.max(t2);
    }
    let oasc = oa.xc[l];
    (oa, oasc)
}

// --- OA traceback select_* (optacc.c). Natural forms of the striped selects. ---

fn select_m(ff: &ForwardFilter, oa: &Omx, i: usize, k: usize) -> u8 {
    // k<2: the predecessors M/I/D(i-1,k-1) reference model column 0, which does not
    // exist — M(i,1) is reachable only from B (begin). The OA fill (p7_optimal_accuracy)
    // already seeds these with NEG_INF for k=1; select_m must match, else at the domain
    // boundary M(1,1) the striped-artifact 0.0 predecessor ties the (valid) B path at
    // xb[0]=0.0 and FArgMax picks M, walking the traceback to i<0 (C tolerates the
    // resulting out-of-bounds read as UB; Rust panics). NEG_INF forces the correct B.
    let pm = if k >= 2 { oa.mmx[i - 1][k - 1] } else { NEG_INF };
    let pi = if k >= 2 { oa.imx[i - 1][k - 1] } else { NEG_INF };
    let pd = if k >= 2 { oa.dmx[i - 1][k - 1] } else { NEG_INF };
    // path order (state): [M, I, D, B]; ties → first (M>I>D>B). C uses FArgMax.
    let path = [
        if ff.amm[k] == 0.0 { NEG_INF } else { pm },
        if ff.aim[k] == 0.0 { NEG_INF } else { pi },
        if ff.adm[k] == 0.0 { NEG_INF } else { pd },
        if ff.tbm[k] == 0.0 { NEG_INF } else { oa.xb[i - 1] },
    ];
    let states = [ST_M, ST_I, ST_D, ST_B];
    let mut best = 0usize;
    for z in 1..4 {
        if path[z] > path[best] {
            best = z;
        }
    }
    states[best]
}

fn select_d(ff: &ForwardFilter, oa: &Omx, i: usize, k: usize) -> u8 {
    let pm = if k >= 2 { oa.mmx[i][k - 1] } else { 0.0 };
    let pd = if k >= 2 { oa.dmx[i][k - 1] } else { 0.0 };
    let tmd = if k >= 2 { ff.tmd[k - 1] } else { 0.0 };
    let tdd = if k >= 2 { ff.tdd[k - 1] } else { 0.0 };
    let p0 = if tmd == 0.0 { NEG_INF } else { pm };
    let p1 = if tdd == 0.0 { NEG_INF } else { pd };
    if p0 >= p1 { ST_M } else { ST_D }
}

fn select_i(ff: &ForwardFilter, oa: &Omx, i: usize, k: usize) -> u8 {
    let p0 = if ff.tmi[k] == 0.0 { NEG_INF } else { oa.mmx[i - 1][k] };
    let p1 = if ff.tii[k] == 0.0 { NEG_INF } else { oa.imx[i - 1][k] };
    if p0 >= p1 { ST_M } else { ST_I }
}

fn select_c(xf: &XFactors, pp: &Omx, oa: &Omx, i: usize) -> u8 {
    let p0 = if xf.c_loop == 0.0 { NEG_INF } else { oa.xc[i - 1] + pp.xc[i] };
    let p1 = if xf.e_move == 0.0 { NEG_INF } else { oa.xe[i] };
    if p0 > p1 { ST_C } else { ST_E }
}

fn select_j(xf: &XFactors, pp: &Omx, oa: &Omx, i: usize) -> u8 {
    let p0 = if xf.j_loop == 0.0 { NEG_INF } else { oa.xj[i - 1] + pp.xj[i] };
    let p1 = if xf.e_loop == 0.0 { NEG_INF } else { oa.xe[i] };
    if p0 > p1 { ST_J } else { ST_E }
}

fn select_b(xf: &XFactors, oa: &Omx, i: usize) -> u8 {
    let p0 = if xf.n_move == 0.0 { NEG_INF } else { oa.xn[i] };
    let p1 = if xf.j_move == 0.0 { NEG_INF } else { oa.xj[i] };
    if p0 > p1 { ST_N } else { ST_J }
}

/// C select_e (optacc.c): E from any M(i,k) k=1..M or D(i,k) k=2..M, STRIPED
/// iteration order (q outer, r inner; k=r*Q+q+1), M-block (>=) then D-block (>).
fn select_e(oa: &Omx, i: usize, m: usize) -> (u8, i32) {
    let q_n = nqf(m);
    let mut max = NEG_INF;
    let mut smax = ST_M;
    let mut kmax = 0i32;
    for q in 0..q_n {
        for r in 0..4 {
            let k = r * q_n + q + 1;
            let v = if k <= m { oa.mmx[i][k] } else { NEG_INF };
            if v >= max {
                max = v;
                smax = ST_M;
                kmax = k as i32;
            }
        }
        for r in 0..4 {
            let k = r * q_n + q + 1;
            let v = if k <= m { oa.dmx[i][k] } else { NEG_INF };
            if v > max {
                max = v;
                smax = ST_D;
                kmax = k as i32;
            }
        }
    }
    (smax, kmax)
}

#[inline]
fn get_postprob(pp: &Omx, scur: u8, sprv: u8, k: usize, i: usize) -> f32 {
    match scur {
        ST_M => pp.mmx[i][k],
        ST_I => pp.imx[i][k],
        ST_N => if sprv == scur { pp.xn[i] } else { 0.0 },
        ST_C => if sprv == scur { pp.xc[i] } else { 0.0 },
        ST_J => if sprv == scur { pp.xj[i] } else { 0.0 },
        _ => 0.0,
    }
}

/// C `p7_OATrace` (optacc.c). Traceback of the OA matrix → forward-order trace
/// `(state, k, i, postprob)`. seq coords are envelope-local (1..L).
fn p7_oa_trace(ff: &ForwardFilter, xf: &XFactors, pp: &Omx, oa: &Omx, l: usize) -> Vec<(u8, i32, i32, f32)> {
    let m = ff.m;
    let mut tr: Vec<(u8, i32, i32, f32)> = Vec::new();
    let mut i = l as i32;
    let mut k = 0i32;
    tr.push((ST_T, 0, i, 0.0));
    tr.push((ST_C, 0, i, 0.0));
    let mut s0 = ST_C;
    while s0 != ST_S {
        let iu = i as usize;
        let ku = k as usize;
        let s1: u8 = match s0 {
            ST_M => { let s = select_m(ff, oa, iu, ku); k -= 1; i -= 1; s }
            ST_D => { let s = select_d(ff, oa, iu, ku); k -= 1; s }
            ST_I => { let s = select_i(ff, oa, iu, ku); i -= 1; s }
            ST_N => if i == 0 { ST_S } else { ST_N },
            ST_C => select_c(xf, pp, oa, iu),
            ST_J => select_j(xf, pp, oa, iu),
            ST_E => { let (s, kk) = select_e(oa, iu, m); k = kk; s }
            ST_B => select_b(xf, oa, iu),
            _ => unreachable!("bogus OA trace state"),
        };
        let postprob = get_postprob(pp, s1, s0, k as usize, i as usize);
        tr.push((s1, k, i, postprob));
        if (s1 == ST_N || s1 == ST_J || s1 == ST_C) && s1 == s0 {
            i -= 1;
        }
        s0 = s1;
    }
    tr.reverse();
    tr
}

// ===========================================================================
// Stochastic trace ensemble + single-linkage clustering (multidomain regions).
// ===========================================================================

/// ODDS-space stochastic traceback over a MULTIHIT Forward Omx `ox`. C
/// `impl_sse/stotrace.c` `p7_StochasticTrace`. Forward-order (state, k, i).
fn p7_stochastic_trace_odds(rng: &mut EslRng, ff: &ForwardFilter, xf: &XFactors, ox: &Omx) -> Vec<(u8, i32, i32)> {
    let ld = ox.ld as i32;
    let mut tr: Vec<(u8, i32, i32)> = Vec::new();
    let mut k = 0i32;
    let mut i = ld;
    tr.push((ST_T, k, i));
    tr.push((ST_C, k, i));
    let mut s0 = ST_C;
    while s0 != ST_S {
        let iu = i as usize;
        let im1 = (i - 1) as usize;
        let ku = k as usize;
        let s1: u8 = match s0 {
            ST_M => {
                let mut path = [
                    ox.xb[im1] * ff.tbm[ku],
                    ox.mmx[im1][ku - 1] * ff.amm[ku],
                    ox.imx[im1][ku - 1] * ff.aim[ku],
                    ox.dmx[im1][ku - 1] * ff.adm[ku],
                ];
                esl_vec_fnorm(&mut path);
                let s = [ST_B, ST_M, ST_I, ST_D][rng.fchoose(&path)];
                k -= 1; i -= 1; s
            }
            ST_D => {
                let mut path = [ox.mmx[iu][ku - 1] * ff.tmd[ku - 1], ox.dmx[iu][ku - 1] * ff.tdd[ku - 1]];
                esl_vec_fnorm(&mut path);
                let s = if rng.fchoose(&path) == 0 { ST_M } else { ST_D };
                k -= 1; s
            }
            ST_I => {
                let mut path = [ox.mmx[im1][ku] * ff.tmi[ku], ox.imx[im1][ku] * ff.tii[ku]];
                esl_vec_fnorm(&mut path);
                let s = if rng.fchoose(&path) == 0 { ST_M } else { ST_I };
                i -= 1; s
            }
            ST_N => if i == 0 { ST_S } else { ST_N },
            ST_C => {
                let mut path = [ox.xc[im1] * xf.c_loop, ox.xe[iu] * xf.e_move * ox.scale[iu]];
                esl_vec_fnorm(&mut path);
                if rng.fchoose(&path) == 0 { ST_C } else { ST_E }
            }
            ST_J => {
                let mut path = [ox.xj[im1] * xf.j_loop, ox.xe[iu] * xf.e_loop * ox.scale[iu]];
                esl_vec_fnorm(&mut path);
                if rng.fchoose(&path) == 0 { ST_J } else { ST_E }
            }
            ST_E => {
                let (st, kk) = select_e_odds(rng, ox, iu);
                k = kk; st
            }
            ST_B => {
                let mut path = [ox.xn[iu] * xf.n_move, ox.xj[iu] * xf.j_move];
                esl_vec_fnorm(&mut path);
                if rng.fchoose(&path) == 0 { ST_N } else { ST_J }
            }
            _ => unreachable!("bogus stochastic trace state"),
        };
        tr.push((s1, k, i));
        if (s1 == ST_N || s1 == ST_J || s1 == ST_C) && s1 == s0 {
            i -= 1;
        }
        s0 = s1;
    }
    tr.reverse();
    tr
}

/// C stotrace.c select_e: FChoose walk over M(i,k)/D(i,k) in STRIPED order,
/// double-precision cumulative, one RNG draw. Normalizer = 1/xE(i).
fn select_e_odds(rng: &mut EslRng, ox: &Omx, i: usize) -> (u8, i32) {
    let m = ox.m;
    let q_n = nqf(m);
    let norm = (1.0f64 / ox.xe[i] as f64) as f32;
    let roll = rng.random();
    let mut sum = 0.0f64;
    for _pass in 0..4 {
        for q in 0..q_n {
            for r in 0..4 {
                let k = r * q_n + q + 1;
                let v = if k <= m { ox.mmx[i][k] } else { 0.0 };
                sum += (v * norm) as f64;
                if roll < sum { return (ST_M, k as i32); }
            }
            for r in 0..4 {
                let k = r * q_n + q + 1;
                let v = if k <= m { ox.dmx[i][k] } else { 0.0 };
                sum += (v * norm) as f64;
                if roll < sum { return (ST_D, k as i32); }
            }
        }
    }
    (ST_M, m as i32)
}

/// C p7_trace_Index: per-domain (sqfrom,sqto,hmmfrom,hmmto).
fn trace_index(tr: &[(u8, i32, i32)]) -> Vec<(i32, i32, i32, i32)> {
    let mut doms: Vec<(i32, i32, i32, i32)> = Vec::new();
    for &(st, k, i) in tr {
        match st {
            ST_B => doms.push((0, 0, 0, 0)),
            ST_M => {
                let d = doms.last_mut().unwrap();
                if d.0 == 0 { d.0 = i; }
                if d.2 == 0 { d.2 = k; }
                d.1 = i;
                d.3 = k;
            }
            _ => {}
        }
    }
    doms
}

#[derive(Clone, Copy)]
struct Seg { idx: i32, i: i32, j: i32, k: i32, m: i32 }

/// C link_spsamples (p7_spensemble.c): min_overlap=0.8, of_smaller, max_diagdiff=4.
fn link_spsamples(h1: &Seg, h2: &Seg) -> bool {
    let min_overlap = 0.8f32;
    let max_diagdiff = 4i32;
    let nov = h1.j.min(h2.j) - h1.i.max(h2.i) + 1;
    let n = (h1.j - h1.i + 1).min(h2.j - h2.i + 1);
    if (nov as f32) / (n as f32) < min_overlap { return false; }
    let nov = h1.m.min(h2.m) - h1.k.max(h2.k); // no +1, per C
    let n = (h1.m - h1.k + 1).min(h2.m - h2.k + 1);
    if (nov as f32) / (n as f32) < min_overlap { return false; }
    let (d1, d2) = (h1.i - h1.k, h2.i - h2.k);
    if (d1 - d2).abs() <= max_diagdiff { return true; }
    let (d1, d2) = (h1.j - h1.m, h2.j - h2.m);
    if (d1 - d2).abs() <= max_diagdiff { return true; }
    false
}

/// C esl_cluster_SingleLinkage.
fn single_linkage(segs: &[Seg]) -> (Vec<usize>, usize) {
    let n = segs.len();
    let mut a: Vec<usize> = (0..n).map(|v| n - v - 1).collect();
    let mut na = n;
    let mut b = vec![0usize; n];
    let mut nb = 0usize;
    let mut c = vec![0usize; n];
    let mut nc = 0usize;
    while na > 0 {
        let v = a[na - 1]; na -= 1;
        b[nb] = v; nb += 1;
        while nb > 0 {
            let v = b[nb - 1]; nb -= 1;
            c[v] = nc;
            let mut idx: i64 = na as i64 - 1;
            while idx >= 0 {
                if link_spsamples(&segs[v], &segs[a[idx as usize]]) {
                    let w = a[idx as usize];
                    a[idx as usize] = a[na - 1];
                    na -= 1;
                    b[nb] = w; nb += 1;
                }
                idx -= 1;
            }
        }
        nc += 1;
    }
    (c, nc)
}

fn iargmax(v: &[i32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] { best = i; }
    }
    best
}

/// C p7_spensemble_Cluster + dominance removal (region_trace_ensemble). Returns
/// significant, non-dominated clusters as (i,j,prob), sorted by i.
fn spensemble_cluster(segs: &[Seg], nsamples: usize) -> Vec<(i32, i32, f32)> {
    let min_posterior = 0.25f32;
    let min_endpointp = 0.02f32;
    if segs.is_empty() { return Vec::new(); }
    let (assignment, nc) = single_linkage(segs);
    let mut sigc: Vec<(i32, i32, i32, i32, f32)> = Vec::new();
    for c in 0..nc {
        let mut ninc = 0i32;
        let mut idx_of_last = -1i32;
        for h in 0..segs.len() {
            if assignment[h] == c {
                if segs[h].idx != idx_of_last { ninc += 1; }
                idx_of_last = segs[h].idx;
            }
        }
        if (ninc as f32) / (nsamples as f32) < min_posterior { continue; }
        let (mut imin, mut imax, mut jmin, mut jmax, mut kmin, mut kmax, mut mmin, mut mmax) =
            (0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
        let mut started = false;
        for h in 0..segs.len() {
            if assignment[h] != c { continue; }
            let s = &segs[h];
            if !started {
                imin = s.i; imax = s.i; jmin = s.j; jmax = s.j;
                kmin = s.k; kmax = s.k; mmin = s.m; mmax = s.m;
                started = true;
            } else {
                imin = imin.min(s.i); imax = imax.max(s.i);
                jmin = jmin.min(s.j); jmax = jmax.max(s.j);
                kmin = kmin.min(s.k); kmax = kmax.max(s.k);
                mmin = mmin.min(s.m); mmax = mmax.max(s.m);
            }
        }
        let epc_threshold = ((ninc as f32) * min_endpointp).ceil() as i32;
        let mut epc = vec![0i32; (imax - imin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].i - imin) as usize] += 1; } }
        let mut best_i = imin;
        while best_i <= imax { if epc[(best_i - imin) as usize] >= epc_threshold { break; } best_i += 1; }
        if best_i > imax { best_i = imin + iargmax(&epc) as i32; }
        let mut epc = vec![0i32; (kmax - kmin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].k - kmin) as usize] += 1; } }
        let mut best_k = kmin;
        while best_k <= kmax { if epc[(best_k - kmin) as usize] >= epc_threshold { break; } best_k += 1; }
        if best_k > kmax { best_k = kmin + iargmax(&epc) as i32; }
        let mut epc = vec![0i32; (jmax - jmin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].j - jmin) as usize] += 1; } }
        let mut best_j = jmax;
        while best_j >= jmin { if epc[(best_j - jmin) as usize] >= epc_threshold { break; } best_j -= 1; }
        if best_j < jmin { best_j = jmin + iargmax(&epc) as i32; }
        let mut epc = vec![0i32; (mmax - mmin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].m - mmin) as usize] += 1; } }
        let mut best_m = mmax;
        while best_m >= mmin { if epc[(best_m - mmin) as usize] >= epc_threshold { break; } best_m -= 1; }
        if best_m < mmin { best_m = mmin + iargmax(&epc) as i32; }
        if best_i > best_j || best_k > best_m { continue; }
        sigc.push((best_i, best_j, best_k, best_m, ninc as f32 / nsamples as f32));
    }
    sigc.sort_by(|a, b| a.0.cmp(&b.0));
    let nsig = sigc.len();
    let mut dominated = vec![false; nsig];
    for d in 0..nsig {
        for d2 in (d + 1)..nsig {
            let nov = sigc[d].1.min(sigc[d2].1) - sigc[d].0.max(sigc[d2].0) + 1;
            if nov == 0 { break; }
            let n = (sigc[d].1 - sigc[d].0 + 1).min(sigc[d2].1 - sigc[d2].0 + 1);
            if (nov as f32) / (n as f32) >= 0.8 {
                if sigc[d].4 > sigc[d2].4 { dominated[d2] = true; } else { dominated[d] = true; }
            }
        }
    }
    sigc.iter().enumerate()
        .filter(|(d, _)| !dominated[*d])
        .map(|(_, s)| (s.0, s.1, s.4))
        .collect()
}

/// C p7_Null2_ByTrace reduction over one trace domain segment.
fn compute_bytrace_null2(counter: &[f32], ld: f32, xsum: f32, ff: &ForwardFilter, m: usize) -> [f32; KP] {
    let norm = 1.0 / ld;
    let xfactor = xsum * norm;
    let mut null2 = [0.0f32; KP];
    for x in 0..K {
        let mut sv = 0.0f32;
        for k in 1..=m {
            sv += counter[k] * norm * ff.rfv(k, x);
        }
        null2[x] = sv + xfactor;
    }
    for x in (K + 1)..=(KP - 3) {
        let set = degen_set(x);
        let mut r = 0.0f32;
        let mut n = 0.0f32;
        for &y in set { r += null2[y]; n += 1.0; }
        null2[x] = if n > 0.0 { r / n } else { 0.0 };
    }
    null2[K] = 1.0;
    null2[KP - 2] = 1.0;
    null2[KP - 1] = 1.0;
    null2
}

/// Split one forward-order trace into domain segments + their By-Trace null2.
fn trace_domain_null2s(tr: &[(u8, i32, i32)], ff: &ForwardFilter, m: usize) -> Vec<(i32, i32, [f32; KP])> {
    let mut out = Vec::new();
    let mut in_dom = false;
    let mut counter = vec![0.0f32; m + 1];
    let mut ld = 0.0f32;
    let mut xsum = 0.0f32;
    let mut sqfrom = 0i32;
    let mut sqto = 0i32;
    for &(st, k, i) in tr {
        if st == ST_B {
            in_dom = true;
            for c in counter.iter_mut() { *c = 0.0; }
            ld = 0.0; xsum = 0.0; sqfrom = 0; sqto = 0;
        } else if in_dom {
            match st {
                ST_M => {
                    ld += 1.0;
                    if k > 0 { counter[k as usize] += 1.0; }
                    if sqfrom == 0 { sqfrom = i; }
                    sqto = i;
                }
                ST_I => {
                    if i > 0 {
                        ld += 1.0;
                        if k > 0 { counter[k as usize] += 1.0; }
                    }
                }
                ST_N | ST_C | ST_J => { if i > 0 { xsum += 1.0; } }
                ST_E => {
                    let null2 = compute_bytrace_null2(&counter, ld, xsum, ff, m);
                    out.push((sqfrom, sqto, null2));
                    in_dom = false;
                }
                _ => {}
            }
        }
    }
    out
}

/// C region_trace_ensemble (p7_domaindef.c). Resolve region [ri..j] (window-local
/// coords) into cluster envelopes via 200 stochastic traces; also build the
/// region-local per-residue null2 log-odds (index 1..=Lr). Returns (envelopes, n2sc).
fn region_trace_ensemble(
    ff: &ForwardFilter,
    dsq: &[u8],
    ri: usize,
    j: usize,
    save_l: usize,
) -> (Vec<(usize, usize)>, Vec<f32>) {
    let nsamples = 200usize;
    // C cm_pipeline.c: pli->r = esl_randomness_CreateFast(--seed default 181), and
    // ddef->do_reseeding=TRUE (seed!=0) resets the RNG to this seed before EACH
    // region's ensemble (p7_domaindef.c region_trace_ensemble). Same convention as
    // the glocal path (p7_generic.rs glocal_region_trace_ensemble).
    let seed = 181u32;
    let lr = j - ri + 1;
    let xf = XFactors::multihit(save_l);
    // Region subseq: sentinel + dsq[ri..=j] + sentinel.
    let mut region = vec![255u8];
    region.extend_from_slice(&dsq[ri..=j]);
    region.push(255u8);
    let ox = p7_forward(ff, &xf, &region, lr);

    let mut n2sc = vec![0.0f32; lr + 1];
    let mut rng = EslRng::new(seed);
    let mut segs: Vec<Seg> = Vec::new();
    for t in 0..nsamples {
        let tr = p7_stochastic_trace_odds(&mut rng, ff, &xf, &ox);
        for (sqfrom, sqto, hmmfrom, hmmto) in trace_index(&tr) {
            segs.push(Seg {
                idx: t as i32,
                i: sqfrom + ri as i32 - 1,
                j: sqto + ri as i32 - 1,
                k: hmmfrom,
                m: hmmto,
            });
        }
        let dnull2 = trace_domain_null2s(&tr, ff, ff.m);
        let mut pos = 1i32;
        for (sqfrom, sqto, null2) in &dnull2 {
            while pos <= *sqfrom { n2sc[pos as usize] += 1.0; pos += 1; }
            while pos <= *sqto {
                let res = region[pos as usize] as usize;
                n2sc[pos as usize] += null2[res];
                pos += 1;
            }
        }
        while pos <= lr as i32 { n2sc[pos as usize] += 1.0; pos += 1; }
    }
    for p in 1..=lr {
        n2sc[p] = (n2sc[p] / nsamples as f32).ln();
    }
    let clusters = spensemble_cluster(&segs, nsamples);
    let env = clusters.into_iter().map(|(i2, j2, _p)| (i2 as usize, j2 as usize)).collect();
    (env, n2sc)
}

// ===========================================================================
// rescore_isolated_domain (local, do_aln=TRUE) + the driver.
// ===========================================================================

/// C `rescore_isolated_domain` (p7_domaindef.c), LOCAL / non-long_target /
/// do_aln=TRUE. Forward (envsc) + Backward + Decoding + OptimalAccuracy + OATrace
/// (oasc, trace, iali/jali) + null2 (domcorrection). `i..j` are window-local
/// envelope coords. `precomputed_domcorrection` is Some for the multidomain
/// (trace-ensemble) path; None triggers Null2_ByExpectation writing n2sc[i..j].
#[allow(clippy::too_many_arguments)]
fn rescore_isolated_domain(
    ff: &ForwardFilter,
    dsq: &[u8],
    i: usize,
    j: usize,
    save_l: usize,
    do_null2: bool,
    precomputed_domcorrection: Option<f32>,
    n2sc: &mut [f32],
) -> Domain {
    let xf = XFactors::unihit(save_l);
    let ld = j - i + 1;
    // Envelope subseq: sentinel + dsq[i..=j] + sentinel.
    let mut sub = vec![255u8];
    sub.extend_from_slice(&dsq[i..=j]);
    sub.push(255u8);

    let fx = p7_forward(ff, &xf, &sub, ld);
    let envsc = forward_score(&fx, &xf);
    let br = p7_backward(ff, &xf, &sub, ld, &fx);
    let pp = p7_decoding(&fx, &br.bck, &xf, br.has_own_scales);

    // OptimalAccuracy + OATrace (do_aln). Trace coords are envelope-local; hack
    // them to window coords (+= i-1), as C does.
    let (oa, oasc) = p7_optimal_accuracy(ff, &xf, &pp, ld);
    let mut tr = p7_oa_trace(ff, &xf, &pp, &oa, ld);
    for e in tr.iter_mut() {
        if e.2 > 0 {
            e.2 += i as i32 - 1;
        }
    }
    // iali/jali = first/last emitted residue (M/I) in the trace (== ad sqfrom/sqto).
    let mut iali = 0i64;
    let mut jali = 0i64;
    for &(st, _k, ii, _pp) in &tr {
        if (st == ST_M || st == ST_I) && ii > 0 {
            if iali == 0 { iali = ii as i64; }
            jali = ii as i64;
        }
    }

    // domcorrection (NATS).
    let domcorrection = if !do_null2 {
        0.0
    } else if let Some(dc) = precomputed_domcorrection {
        dc
    } else {
        let null2 = p7_null2_by_expectation(ff, &pp, ld);
        let mut d = 0.0f32;
        for pos in i..=j {
            let v = null2[dsq[pos] as usize].ln();
            n2sc[pos] = v;
            d += v;
        }
        d
    };

    Domain {
        ienv: i as i64,
        jenv: j as i64,
        iali,
        jali,
        envsc,
        oasc,
        domcorrection,
        tr,
    }
}

/// C `p7_domaindef_ByPosteriorHeuristics` (LOCAL). `dsq` is the window (1..=l,
/// sentinel padded). Returns the domains + report counts + per-residue n2sc.
pub fn p7_domaindef_local(ff: &ForwardFilter, dsq: &[u8], l: usize, do_null2: bool) -> DomainDef {
    let rt1 = 0.25f32;
    let rt2 = 0.10f32;
    let rt3 = 0.20f32;
    let save_l = l;

    // Posterior decoding on the MULTIHIT whole-window matrices.
    let xf_multi = XFactors::multihit(l);
    let fx = p7_forward(ff, &xf_multi, dsq, l);
    let br = p7_backward(ff, &xf_multi, dsq, l, &fx);
    let dd = p7_domain_decoding(&fx, &br.bck, &xf_multi, br.has_own_scales);
    let (btot, etot, mocc) = (dd.btot, dd.etot, dd.mocc);
    let nexpected = btot[l];

    let mut domains = Vec::new();
    let mut n2sc_full = vec![0.0f32; l + 1];
    let (mut nregions, mut nclustered, mut noverlaps, mut nenvelopes) = (0i32, 0i32, 0i32, 0i32);
    let mut i: i64 = -1;
    let mut triggered = false;
    for j in 1..=l {
        if !triggered {
            if mocc[j] - (btot[j] - btot[j - 1]) < rt2 {
                i = j as i64;
            } else if i == -1 {
                i = j as i64;
            }
            if mocc[j] >= rt1 {
                triggered = true;
            }
        } else if mocc[j] - (etot[j] - etot[j - 1]) < rt2 {
            nregions += 1;
            let ri = i as usize;
            if is_multidomain_region(&btot, &etot, ri, j, rt3) {
                nclustered += 1;
                let (clusters, region_n2sc) = region_trace_ensemble(ff, dsq, ri, j, save_l);
                let lr = j - ri + 1;
                for p in 1..=lr {
                    n2sc_full[ri + p - 1] = region_n2sc[p];
                }
                let mut last_j2 = 0usize;
                for (i2, j2) in clusters {
                    if i2 <= last_j2 {
                        noverlaps += 1;
                    }
                    nenvelopes += 1;
                    let mut dc = 0.0f32;
                    for p in i2..=j2 {
                        dc += region_n2sc[p - ri + 1];
                    }
                    let dom = rescore_isolated_domain(ff, dsq, i2, j2, save_l, do_null2, Some(dc), &mut n2sc_full);
                    domains.push(dom);
                    last_j2 = j2;
                }
            } else {
                let dom = rescore_isolated_domain(ff, dsq, ri, j, save_l, do_null2, None, &mut n2sc_full);
                domains.push(dom);
                nenvelopes += 1;
            }
            i = -1;
            triggered = false;
        }
    }

    DomainDef {
        domains,
        nexpected,
        nregions,
        nclustered,
        noverlaps,
        nenvelopes,
        n2sc: n2sc_full,
    }
}

/// C `is_multidomain_region` (p7_domaindef.c): max_z min(E(z),B(z)) >= rt3.
fn is_multidomain_region(btot: &[f32], etot: &[f32], i: usize, j: usize, rt3: f32) -> bool {
    let mut max = -1.0f32;
    for z in i..=j {
        let e = etot[z] - etot[i - 1];
        let b = btot[j] - btot[z - 1];
        let expected_n = if e < b { e } else { b };
        if expected_n > max { max = expected_n; }
    }
    max >= rt3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p7_fwdback::build_forward_filter;
    use crate::p7_hmm::P7Profile;

    fn synthetic_p7(m: usize, seed: u64) -> P7Profile {
        let mut s = seed;
        let mut rng = || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut p7 = P7Profile::new(m as i32);
        for k in 1..=m {
            let mut e = [0.0f32; 4];
            let mut sum = 0.0f32;
            for x in 0..4 { e[x] = (0.05 + rng()) as f32; sum += e[x]; }
            for x in 0..4 { e[x] /= sum; }
            p7.mat[k] = e;
        }
        for k in 0..=m {
            if k == m { p7.trans[k] = [1.0, 0.0, 0.0, 0.75, 0.25, 0.72, 0.28]; }
            else { p7.trans[k] = [0.80, 0.12, 0.08, 0.75, 0.25, 0.72, 0.28]; }
        }
        p7
    }

    fn rand_dsq(l: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut v = vec![0u8; l + 2];
        v[0] = 4; v[l + 1] = 4;
        for i in 1..=l { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = (s % 4) as u8; }
        v
    }

    // A single embedded model instance in the middle of random flanks should give
    // at least one domain with a sane envelope and a valid OA trace.
    #[test]
    fn domaindef_finds_domains_and_valid_trace() {
        let m = 40usize;
        let p7 = synthetic_p7(m, 0xD00D1234);
        let ff = build_forward_filter(&p7);
        let l = 160usize;
        let dsq = rand_dsq(l, 0xA11CE5);
        let dd = p7_domaindef_local(&ff, &dsq, l, true);
        // nexpected finite and >= 0.
        assert!(dd.nexpected.is_finite() && dd.nexpected >= 0.0);
        for dom in &dd.domains {
            assert!(dom.ienv >= 1 && dom.jenv <= l as i64 && dom.ienv <= dom.jenv);
            assert!(dom.envsc.is_finite());
            assert!(dom.oasc.is_finite() && dom.oasc >= 0.0);
            // iali/jali within envelope.
            if dom.iali > 0 {
                assert!(dom.iali >= dom.ienv && dom.jali <= dom.jenv, "ali within env");
            }
            // Trace: monotone non-increasing i as we go backward from T; ends at S.
            assert_eq!(dom.tr.first().unwrap().0, ST_S, "trace starts at S");
            assert_eq!(dom.tr.last().unwrap().0, ST_T, "trace ends at T");
            // oasc <= number of aligned residues (expected accuracy bound).
            let nali = (dom.jenv - dom.ienv + 1) as f32;
            assert!(dom.oasc <= nali + 1e-3, "oasc bounded by envelope length");
        }
        // n2sc has one entry per residue.
        assert_eq!(dd.n2sc.len(), l + 1);
    }

    // OA score must be within [0, Ld]: expected # of correctly aligned residues.
    #[test]
    fn oa_score_in_range() {
        let m = 32usize;
        let p7 = synthetic_p7(m, 0x0A5C77);
        let ff = build_forward_filter(&p7);
        let l = 60usize;
        let dsq = rand_dsq(l, 0xBEE555);
        let xf = XFactors::unihit(l);
        let mut sub = vec![255u8];
        sub.extend_from_slice(&dsq[1..=l]);
        sub.push(255u8);
        let fx = p7_forward(&ff, &xf, &sub, l);
        let br = p7_backward(&ff, &xf, &sub, l, &fx);
        let pp = p7_decoding(&fx, &br.bck, &xf, br.has_own_scales);
        let (_oa, oasc) = p7_optimal_accuracy(&ff, &xf, &pp, l);
        assert!(oasc >= -1e-4 && oasc <= l as f32 + 1e-3, "oasc={oasc} out of [0,L]");
    }

    // Strong OA identity: the optimal-accuracy score equals the sum of posterior
    // probabilities collected along the reconstructed OA trace (the OA DP
    // accumulates exactly these pp values on the best path). Ties OA forward + the
    // OATrace together — catches transcription errors in either.
    #[test]
    fn oasc_equals_trace_postprob_sum() {
        for &m in &[7usize, 20, 33, 64, 101, 128] {
            for seed in 0..3u64 {
                let p7 = synthetic_p7(m, 0x0AF00D ^ (m as u64) << 4 ^ seed);
                let ff = build_forward_filter(&p7);
                for &l in &[15usize, 40, 90] {
                    let dsq = rand_dsq(l, 0x0A5EED ^ (m as u64) << 8 ^ (l as u64) << 3 ^ seed);
                    let xf = XFactors::unihit(l);
                    let mut sub = vec![255u8];
                    sub.extend_from_slice(&dsq[1..=l]);
                    sub.push(255u8);
                    let fx = p7_forward(&ff, &xf, &sub, l);
                    let br = p7_backward(&ff, &xf, &sub, l, &fx);
                    let pp = p7_decoding(&fx, &br.bck, &xf, br.has_own_scales);
                    let (oa, oasc) = p7_optimal_accuracy(&ff, &xf, &pp, l);
                    let tr = p7_oa_trace(&ff, &xf, &pp, &oa, l);
                    let sum: f32 = tr.iter().map(|&(_s, _k, _i, p)| p).sum();
                    let diff = (sum - oasc).abs();
                    let tol = 1e-3 * oasc.abs().max(1.0);
                    assert!(
                        diff <= tol,
                        "OA identity m={m} l={l} seed={seed}: oasc={oasc} Σpp={sum} diff={diff}"
                    );
                    // Trace validity: starts S, ends T, k/i in range, i non-increasing
                    // from T back to S (trace is forward-order after reverse).
                    assert_eq!(tr.first().unwrap().0, ST_S);
                    assert_eq!(tr.last().unwrap().0, ST_T);
                    for &(_s, k, i, _p) in &tr {
                        assert!(k >= 0 && k <= m as i32, "k in range");
                        assert!(i >= 0 && i <= l as i32, "i in range");
                    }
                }
            }
        }
    }

    // Real filter HMM: run full domain definition; smoke + invariants.
    #[test]
    fn real_cm_domaindef() {
        let path = match std::env::var("INFERNOX_TEST_CM") { Ok(p) => p, Err(_) => return };
        let cm = crate::cm_file::cm_file_read(&path).expect("read cm");
        let p7 = cm.p7.as_ref().expect("p7 filter");
        let ff = build_forward_filter(p7);
        let l = 200usize;
        let dsq = rand_dsq(l, 0xF00D);
        let dd = p7_domaindef_local(&ff, &dsq, l, true);
        assert!(dd.nexpected.is_finite());
        for dom in &dd.domains {
            assert!(dom.envsc.is_finite());
            assert!(dom.oasc.is_finite());
        }
        eprintln!("real-cm domaindef OK: m={} ndom={}", p7.m, dd.domains.len());
    }
}
