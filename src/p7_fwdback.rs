// SPDX-License-Identifier: BSD-3-Clause
//! p7_fwdback — full-matrix odds-space Forward (and, subsequently, Backward /
//! Decoding) for the LOCAL p7 pipeline that `cmsearch --hmmonly` needs.
//!
//! This is a de-striped scalar transcription of HMMER3's SSE
//! `impl_sse/fwdback.c:forward_engine(do_full=TRUE)` (= `p7_Forward`; the
//! score-only `p7_ForwardParser` is the same engine with `do_full=FALSE`). The
//! per-cell M and I recurrences are sets of independent float ops, so they are
//! bit-identical to the striped SSE regardless of striping. Two steps DO depend
//! on the SIMD lane layout and are reproduced exactly:
//!   1. the DD (delete->delete) path: the striped `rsz` cross-slot right-shift
//!      couples with the m>=100 lazy-convergence early-exit break, so the
//!      striped result can differ by 1 ULP from a natural serial fixpoint sweep.
//!      We run the DD in the striped quad/lane layout (position k = z*Q+qi+1)
//!      and scatter D back to natural order.
//!   2. the horizontal `xE = Σ M(i,k) + Σ D(i,k)` reduction (4 lane accumulators
//!      over the striped position map, then a pairwise tree horizontal sum).
//! The full matrix (all rows + per-row scale + specials) is retained for
//! Backward/Decoding.
//!
//! Faithfulness anchors:
//!   * C `impl_sse/fwdback.c:256-463` (forward_engine),
//!     `impl_sse/p7_omx.[ch]` (P7_OMX layout), `impl_sse/io.c`/`p7_oprofile.c`
//!     (`fb_conversion`, the odds-space profile), `modelconfig.c`
//!     (`p7_ProfileConfig` local multihit, `p7_ReconfigLength`).
//!   * Cross-checked against the proven byte-parity de-striped reference in
//!     `rustyhmmer-dev/src/omx.rs::forward` + `src/forward.rs` (same recurrence).
//!   * The Forward SCORE is anchored bit-for-bit against the crate's existing
//!     C-verified striped `cm_pipeline::forward_filter_score` (see tests).
//!
//! The packed-profile builder here is REPLICATED (natural-order arrays only —
//! the de-striped engine does not need the striped `tmain/tddv/rfvv`) so this
//! module shares no source surface with `cm_pipeline.rs`.

use crate::p7_hmm::P7Profile;

pub(crate) const KP: usize = 18; // RNA extended alphabet size (A,C,G,U,gap,degens,*,~)
pub(crate) const K_CANON: usize = 4;

/// Degenerate-code → canonical residue set (RNA). C `esl_abc` degeneracy map;
/// used to average canonical match scores for degenerate codes (uniform bg).
pub(crate) fn degen_set(code: usize) -> &'static [usize] {
    match code {
        0 => &[0],
        1 => &[1],
        2 => &[2],
        3 => &[3],
        5 => &[0, 2],        // R = A|G
        6 => &[1, 3],        // Y = C|U
        7 => &[0, 1],        // M = A|C
        8 => &[2, 3],        // K = G|U
        9 => &[1, 2],        // S = C|G
        10 => &[0, 3],       // W = A|U
        11 => &[0, 1, 3],    // H = A|C|U
        12 => &[1, 2, 3],    // B = C|G|U
        13 => &[0, 1, 2],    // V = A|C|G
        14 => &[0, 2, 3],    // D = A|G|U
        15 => &[0, 1, 2, 3], // N = any
        _ => &[],
    }
}

/// Odds-space p7 profile (natural order) for the local-multihit Forward/Backward.
/// Mirrors `P7_OPROFILE`'s `rfv`/`tfv` after `fb_conversion`, minus the striped
/// packing (the de-striped engine reads these natural arrays directly).
pub struct ForwardFilter {
    pub m: usize,
    /// Transposed emission odds: `rfv_t[x*(m+1) + k] = rfv[k][x] = exp(MSC[k][x])`.
    pub(crate) rfv_t: Vec<f32>,
    /// Local entry `B->M_k = occ[k]/Z` (round-tripped through log/exp), k=1..=M.
    pub(crate) tbm: Vec<f32>,
    /// Transitions INTO M_k from node k-1: amm=M->M, aim=I->M, adm=D->M. k=1..=M.
    pub(crate) amm: Vec<f32>,
    pub(crate) aim: Vec<f32>,
    pub(crate) adm: Vec<f32>,
    /// Insert transitions at node k: tmi=M_k->I_k, tii=I_k->I_k. 0 at k=M.
    pub(crate) tmi: Vec<f32>,
    pub(crate) tii: Vec<f32>,
    /// Delete transitions out of node k: tmd=M_k->D_{k+1}, tdd=D_k->D_{k+1}. 0 at k=M.
    pub(crate) tmd: Vec<f32>,
    pub(crate) tdd: Vec<f32>,
}

impl ForwardFilter {
    /// Emission odds `rfv[k][x] = exp(MSC[k][x])` (k=0..=m match node, x alphabet code).
    #[inline]
    pub(crate) fn rfv(&self, k: usize, x: usize) -> f32 {
        self.rfv_t[x * (self.m + 1) + k]
    }
}

/// Build the odds-space Forward/Backward profile from the p7 filter HMM. Faithful
/// to `p7_ProfileConfig` (LOCAL multihit) + `fb_conversion` (p7_oprofile.c:939-990),
/// natural-order only. This is a value-for-value replica of the natural-order
/// arrays in `cm_pipeline::build_forward_filter` (which the crate's C-verified
/// striped Forward is built from) — so both engines see identical operands.
pub fn build_forward_filter(p7: &P7Profile) -> ForwardFilter {
    let m = p7.m as usize;

    // Match emission scores MSC[k][x] then rfv=exp(MSC). modelconfig.c:141-151
    // (uniform bg f[x]=1/K=0.25): sc[x]=log(mat[k][x]/0.25); degenerates =
    // esl_abc_FExpectScVec = mean of canonical scores in the code's set.
    // fb_conversion (p7_oprofile.c:960): om->rfv[x][q] = expf(MSC).
    let mut rfv = vec![[0.0f32; KP]; m + 1];
    for k in 1..=m {
        let mut sc = [f32::NEG_INFINITY; KP];
        for x in 0..K_CANON {
            sc[x] = ((p7.mat[k][x] as f64) / 0.25).ln() as f32;
        }
        for code in K_CANON..KP {
            let set = degen_set(code);
            if set.is_empty() {
                continue;
            }
            let mut s = 0.0f32;
            for &x in set {
                s += sc[x];
            }
            sc[code] = s / set.len() as f32;
        }
        for x in 0..KP {
            rfv[k][x] = sc[x].exp(); // exp(-inf) = 0
        }
    }
    // Transposed copy: rfv_t[x*(m+1)+k] = rfv[k][x].
    let mut rfv_t = vec![0.0f32; KP * (m + 1)];
    for k in 0..=m {
        for x in 0..KP {
            rfv_t[x * (m + 1) + k] = rfv[k][x];
        }
    }

    // Occupancy mocc[k] — p7_hmm.c::p7_hmm_CalculateOccupancy
    // (trans layout [MM=0,MI=1,MD=2,IM=3,II=4,DM=5,DD=6]):
    //   occ[1] = t[0][MI] + t[0][MM];
    //   occ[k] = occ[k-1]*(t[k-1][MM]+t[k-1][MI]) + (1-occ[k-1])*t[k-1][DM].
    let mut occ = vec![0.0f32; m + 1];
    if m >= 1 {
        occ[1] = p7.trans[0][1] + p7.trans[0][0];
    }
    for k in 2..=m {
        occ[k] = occ[k - 1] * (p7.trans[k - 1][0] + p7.trans[k - 1][1])
            + (1.0 - occ[k - 1]) * p7.trans[k - 1][5];
    }
    // Local entry (modelconfig.c:90-97, IsLocal): Z=Σ occ[k]*(M-k+1);
    // tBM[k]=exp(log(occ[k]/Z)) — round-trips through log/exp exactly as C does.
    let mut z = 0.0f32;
    for k in 1..=m {
        z += occ[k] * (m - k + 1) as f32;
    }
    let mut tbm = vec![0.0f32; m + 1];
    for k in 1..=m {
        tbm[k] = ((occ[k] / z) as f64).ln().exp() as f32;
    }

    // Transitions into M_k use node k-1 (fb_conversion kb=k-1 for BM/MM/IM/DM).
    let mut amm = vec![0.0f32; m + 1];
    let mut aim = vec![0.0f32; m + 1];
    let mut adm = vec![0.0f32; m + 1];
    for k in 1..=m {
        amm[k] = p7.trans[k - 1][0]; // M_{k-1}->M_k
        aim[k] = p7.trans[k - 1][3]; // I_{k-1}->M_k
        adm[k] = p7.trans[k - 1][5]; // D_{k-1}->M_k
    }
    // Insert / delete-out at node k; impossible (0) at k=M (fb_conversion's
    // kb+z*nq < M test is false there).
    let mut tmi = vec![0.0f32; m + 1];
    let mut tii = vec![0.0f32; m + 1];
    let mut tmd = vec![0.0f32; m + 1];
    let mut tdd = vec![0.0f32; m + 1];
    for k in 1..m {
        tmi[k] = p7.trans[k][1]; // M_k->I_k
        tii[k] = p7.trans[k][4]; // I_k->I_k
        tmd[k] = p7.trans[k][2]; // M_k->D_{k+1}
        tdd[k] = p7.trans[k][6]; // D_k->D_{k+1}
    }

    ForwardFilter {
        m,
        rfv_t,
        tbm,
        amm,
        aim,
        adm,
        tmi,
        tii,
        tmd,
        tdd,
    }
}

/// Odds-space special-state length model (the SSE `om->xf` table, LINEAR probs).
/// C `p7_oprofile_ReconfigLength` + the E-state factors (multihit vs unihit).
pub struct XFactors {
    pub n_loop: f32,
    pub n_move: f32,
    pub c_loop: f32,
    pub c_move: f32,
    pub j_loop: f32,
    pub j_move: f32,
    pub e_move: f32,
    pub e_loop: f32,
}

impl XFactors {
    /// MULTIHIT local length model for full sequence length `l`. C
    /// `p7_oprofile_ReconfigMultihit`/`p7_ReconfigLength` (modelconfig.c:228-231,
    /// nj=1): pmove = 3/(L+3), ploop = 1-pmove for N/C/J; E: MOVE=LOOP=0.5
    /// (exp(-log2), modelconfig.c:116-117). This matches the crate's C-verified
    /// `forward_filter_score`.
    pub fn multihit(l: usize) -> Self {
        let nj = 1.0f32;
        let pmove = (2.0 + nj) / (l as f32 + 2.0 + nj);
        let ploop = 1.0 - pmove;
        XFactors {
            n_loop: ploop,
            n_move: pmove,
            c_loop: ploop,
            c_move: pmove,
            j_loop: ploop,
            j_move: pmove,
            e_move: 0.5,
            e_loop: 0.5,
        }
    }

    /// UNIHIT local length model for full sequence length `l`. C
    /// `p7_oprofile_ReconfigUnihit` (nj=0): pmove = 2/(L+2), ploop = 1-pmove for
    /// N/C/J; E: MOVE=1.0 (E->C only), LOOP=0.0 (no J loop). Used for the
    /// per-domain rescore in domain definition.
    pub fn unihit(l: usize) -> Self {
        let nj = 0.0f32;
        let pmove = (2.0 + nj) / (l as f32 + 2.0 + nj);
        let ploop = 1.0 - pmove;
        XFactors {
            n_loop: ploop,
            n_move: pmove,
            c_loop: ploop,
            c_move: pmove,
            j_loop: ploop,
            j_move: pmove,
            e_move: 1.0,
            e_loop: 0.0,
        }
    }
}

/// Odds-space Forward/Backward DP matrix (de-striped full form). Rows `0..=ld`,
/// columns `k=0..=m` for M/I/D; specials + per-row SCALE per row. Mirrors
/// `P7_OMX` (`dpf`/`xmx`), minus striping. `totscale` = Σ ln(scale[i]) (C
/// `ox->totscale`).
pub struct Omx {
    pub m: usize,
    pub ld: usize,
    pub mmx: Vec<Vec<f32>>, // [row][k]
    pub imx: Vec<Vec<f32>>,
    pub dmx: Vec<Vec<f32>>,
    pub xe: Vec<f32>,
    pub xn: Vec<f32>,
    pub xj: Vec<f32>,
    pub xb: Vec<f32>,
    pub xc: Vec<f32>,
    pub scale: Vec<f32>,
    pub totscale: f32,
}

impl Omx {
    pub(crate) fn new(m: usize, ld: usize) -> Self {
        Omx {
            m,
            ld,
            mmx: vec![vec![0.0f32; m + 1]; ld + 1],
            imx: vec![vec![0.0f32; m + 1]; ld + 1],
            dmx: vec![vec![0.0f32; m + 1]; ld + 1],
            xe: vec![0.0f32; ld + 1],
            xn: vec![0.0f32; ld + 1],
            xj: vec![0.0f32; ld + 1],
            xb: vec![0.0f32; ld + 1],
            xc: vec![0.0f32; ld + 1],
            scale: vec![1.0f32; ld + 1],
            totscale: 0.0,
        }
    }
}

/// p7O_NQF(M): number of SSE quads = ESL_MAX(2, (M-1)/4 + 1). Determines the
/// striped lane→position map used by the xE horizontal reduction.
#[inline]
pub(crate) fn nqf(m: usize) -> usize {
    if m >= 1 {
        std::cmp::max(2, (m - 1) / 4 + 1)
    } else {
        2
    }
}

/// Full-matrix odds-space Forward. Scalar transcription of
/// `impl_sse/fwdback.c:256-463` `forward_engine(do_full=TRUE)`. `dsq` is
/// 1-indexed with sentinels; residues `dsq[1..=l]` are alphabet codes in 0..KP.
/// Returns the retained `Omx` (score via [`forward_score`]).
pub fn p7_forward(ff: &ForwardFilter, xf: &XFactors, dsq: &[u8], l: usize) -> Omx {
    let m = ff.m;
    let mut ox = Omx::new(m, l);
    let q_n = nqf(m);

    // fwdback.c:279-288 — zero row 0; xN=1, xB=xf[N][MOVE], xE=xJ=xC=0; SCALE=1.
    ox.xe[0] = 0.0;
    ox.xn[0] = 1.0;
    ox.xj[0] = 0.0;
    ox.xb[0] = xf.n_move;
    ox.xc[0] = 0.0;
    ox.scale[0] = 1.0;

    let mut xn = 1.0f32;
    let mut xj = 0.0f32;
    let mut xb = xf.n_move;
    let mut xc = 0.0f32;

    // Rolling previous-row cell vectors (odds space; index 0 always 0).
    let mut mp = vec![0.0f32; m + 1];
    let mut ip = vec![0.0f32; m + 1];
    let mut dp = vec![0.0f32; m + 1];

    for i in 1..=l {
        let x = dsq[i] as usize;
        let mc = &mut ox.mmx[i];
        let ic = &mut ox.imx[i];
        let dc = &mut ox.dmx[i];
        mc[0] = 0.0;
        ic[0] = 0.0;
        dc[0] = 0.0;

        // fwdback.c:309-338 — M and I from the previous row (per-cell independent
        // float ops → bit-identical to the striped lanes; the term order
        // [B->M, M->M, I->M, D->M] then *emission matches the SSE tp sequence).
        {
            let base = x * (m + 1);
            let rfv_row = &ff.rfv_t[base + 1..base + m + 1]; // rfv[k][x], k=1..=m
            let mpm = &mp[0..m]; // mp[k-1]
            let ipm = &ip[0..m];
            let dpm = &dp[0..m];
            let tbm = &ff.tbm[1..m + 1];
            let amm = &ff.amm[1..m + 1];
            let aim = &ff.aim[1..m + 1];
            let adm = &ff.adm[1..m + 1];
            let mc_o = &mut mc[1..m + 1];
            for j in 0..m {
                let sv = xb * tbm[j] + mpm[j] * amm[j] + ipm[j] * aim[j] + dpm[j] * adm[j];
                mc_o[j] = sv * rfv_row[j];
            }
        }
        {
            let mpk = &mp[1..m + 1];
            let ipk = &ip[1..m + 1];
            let tmi = &ff.tmi[1..m + 1];
            let tii = &ff.tii[1..m + 1];
            let ic_o = &mut ic[1..m + 1];
            for j in 0..m {
                ic_o[j] = mpk[j] * tmi[j] + ipk[j] * tii[j];
            }
        }

        // fwdback.c:340-397 — DD paths. These MUST be computed in the striped
        // (Farrar) SIMD lane layout, not a natural serial sweep: the cross-slot
        // right-shift `rsz` couples with the m>=100 lazy-convergence early-exit
        // (`_mm_cmpgt_ps`/`_mm_movemask_ps` break) such that the striped result
        // can differ from the exact natural fixpoint by 1 ULP. A natural sweep
        // matches only the m<100 (fully-serialized 4-pass) branch. We therefore
        // replicate the exact striped DD here (quad qi, lane z <-> position
        // k = z*Q + qi + 1) and scatter D back to natural dc[k]. The M/I cells,
        // xE reduction, specials and rescaling stay de-striped (bit-identical).
        // C impl_sse/fwdback.c:340-397 ; mirrors cm_pipeline forward_filter_score_striped.
        {
            let qn = q_n;
            // rsz: _mm_slli_si128::<4> — lane z gets old z-1, lane 0 -> 0.
            #[inline(always)]
            fn rsz(a: [f32; 4]) -> [f32; 4] {
                [0.0, a[0], a[1], a[2]]
            }
            #[inline(always)]
            fn at(nat: &[f32], k: usize, m: usize) -> f32 {
                if k >= 1 && k <= m { nat[k] } else { 0.0 }
            }
            // Striped D scratch: dmp[qi] holds 4 lanes for quad qi.
            let mut dmp = vec![[0.0f32; 4]; qn];
            // Replicate the main q-loop's delayed D store: dmp[qi] = dcv (prev
            // quad's M->D partial), then dcv = Mvec[qi] * MDvec[qi]. dcv starts 0.
            let mut dcv = [0.0f32; 4];
            for qi in 0..qn {
                dmp[qi] = dcv;
                let mut nd = [0.0f32; 4];
                for z in 0..4 {
                    let k = z * qn + qi + 1;
                    nd[z] = at(mc, k, m) * at(&ff.tmd, k, m);
                }
                dcv = nd;
            }
            // First obligatory DD pass (fwdback.c:349-356): dcv=rsz(dcv); dmp[0]=0.
            dcv = rsz(dcv);
            dmp[0] = [0.0; 4];
            for qi in 0..qn {
                let mut d = [0.0f32; 4];
                for z in 0..4 {
                    d[z] = dcv[z] + dmp[qi][z];
                }
                dmp[qi] = d;
                // dcv = d * tdd  (NOTE: first pass uses the NEW d).
                for z in 0..4 {
                    let k = z * qn + qi + 1;
                    dcv[z] = d[z] * at(&ff.tdd, k, m);
                }
            }
            if m < 100 {
                // Fully serialized: 3 more passes (fwdback.c:366-378).
                for _j in 1..4 {
                    dcv = rsz(dcv);
                    for qi in 0..qn {
                        let mut d = [0.0f32; 4];
                        for z in 0..4 {
                            d[z] = dcv[z] + dmp[qi][z];
                        }
                        dmp[qi] = d;
                        // dcv = dcv * tdd  (subsequent passes use OLD dcv).
                        for z in 0..4 {
                            let k = z * qn + qi + 1;
                            dcv[z] = dcv[z] * at(&ff.tdd, k, m);
                        }
                    }
                }
            } else {
                // Lazy convergence with early-exit break (fwdback.c:379-397).
                for _j in 1..4 {
                    let mut changed = false;
                    dcv = rsz(dcv);
                    for qi in 0..qn {
                        let old = dmp[qi];
                        let mut d = [0.0f32; 4];
                        for z in 0..4 {
                            d[z] = dcv[z] + old[z];
                            if d[z] > old[z] {
                                changed = true; // _mm_cmpgt_ps | into cv
                            }
                        }
                        dmp[qi] = d;
                        for z in 0..4 {
                            let k = z * qn + qi + 1;
                            dcv[z] = dcv[z] * at(&ff.tdd, k, m);
                        }
                    }
                    if !changed {
                        break; // _mm_movemask_ps(cv) == 0
                    }
                }
            }
            // Scatter striped D back to natural dc[k], k = z*Q + qi + 1.
            dc[0] = 0.0;
            for qi in 0..qn {
                for z in 0..4 {
                    let k = z * qn + qi + 1;
                    if k <= m {
                        dc[k] = dmp[qi][z];
                    }
                }
            }
        }

        // fwdback.c:399-409 — xE = Σ M(i,k) + Σ D(i,k). Reproduce the SSE
        // ForwardParser reduction exactly: 4 lane accumulators (lane r sums the
        // contiguous striped block k = r·Q+1..r·Q+Q, q inner) — M-block for all
        // lanes then D-block — then the pairwise TREE horizontal sum
        // (l0+l1)+(l2+l3), NOT sequential l0+l1+l2+l3.
        let mut lane = [0.0f32; 4];
        for (r, l_acc) in lane.iter_mut().enumerate() {
            for q in 0..q_n {
                let k = r * q_n + q + 1;
                if k <= m {
                    *l_acc += mc[k];
                }
            }
        }
        for (r, l_acc) in lane.iter_mut().enumerate() {
            for q in 0..q_n {
                let k = r * q_n + q + 1;
                if k <= m {
                    *l_acc += dc[k];
                }
            }
        }
        let mut xe = (lane[0] + lane[1]) + (lane[2] + lane[3]);

        // fwdback.c:411-414 — specials.
        xn *= xf.n_loop;
        xc = (xc * xf.c_loop) + (xe * xf.e_move);
        xj = (xj * xf.j_loop) + (xe * xf.e_loop);
        xb = (xj * xf.j_move) + (xn * xf.n_move);

        // fwdback.c:417-435 — sparse rescaling when xE > 1e4.
        if xe > 1.0e4 {
            let inv = 1.0 / xe;
            xn *= inv;
            xc *= inv;
            xj *= inv;
            xb *= inv;
            for k in 0..=m {
                mc[k] *= inv;
                dc[k] *= inv;
                ic[k] *= inv;
            }
            ox.scale[i] = xe;
            // ox->totscale += log(xE)  (float accumulation; ln in f64, cast f32).
            ox.totscale += (xe as f64).ln() as f32;
            xe = 1.0;
        } else {
            ox.scale[i] = 1.0;
        }

        // fwdback.c:441-445 — store specials.
        ox.xe[i] = xe;
        ox.xn[i] = xn;
        ox.xj[i] = xj;
        ox.xb[i] = xb;
        ox.xc[i] = xc;

        // Roll current row into "previous".
        mp.copy_from_slice(&ox.mmx[i]);
        ip.copy_from_slice(&ox.imx[i]);
        dp.copy_from_slice(&ox.dmx[i]);
    }

    ox
}

/// Forward lod score in NATS. C `fwdback.c:461`:
/// `*opt_sc = ox->totscale + log(xC * om->xf[p7O_C][p7O_MOVE])`. Matches the
/// crate's striped `forward_filter_score` return
/// (`totscale as f64 + ((xc as f64)*(c_move as f64)).ln()`).
pub fn forward_score(ox: &Omx, xf: &XFactors) -> f32 {
    let xc = ox.xc[ox.ld];
    (ox.totscale as f64 + ((xc as f64) * (xf.c_move as f64)).ln()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small but valid synthetic RNA p7 filter HMM (probabilities normalized
    /// per state) for exercising the DP against the crate's C-verified engine.
    fn synthetic_p7(m: usize, seed: u64) -> P7Profile {
        let mut s = seed;
        let mut rng = || {
            // xorshift64*, just for varied-but-deterministic test data.
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
        // trans[k] = [MM,MI,MD,IM,II,DM,DD], each of the 3 out-distributions
        // (M:{MM,MI,MD}, I:{IM,II}, D:{DM,DD}) normalized. Node M gets M->E only.
        for k in 0..=m {
            let (mm, mi, md) = (0.80f32, 0.12f32, 0.08f32);
            let (im, ii) = (0.75f32, 0.25f32);
            let (dm, dd) = (0.72f32, 0.28f32);
            if k == m {
                p7.trans[k] = [1.0, 0.0, 0.0, im, ii, dm, dd];
            } else {
                p7.trans[k] = [mm, mi, md, im, ii, dm, dd];
            }
        }
        p7
    }

    /// Deterministic pseudo-random digitized RNA sequence (codes 0..=3), 1-indexed
    /// with sentinel guards at 0 and l+1.
    fn rand_dsq(l: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut v = vec![0u8; l + 2];
        v[0] = 4; // eslDSQ_SENTINEL-ish guard (unused by DP)
        v[l + 1] = 4;
        for i in 1..=l {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v[i] = (s % 4) as u8;
        }
        v
    }

    // The Forward SCORE from this de-striped full-matrix engine must be
    // bit-identical to the crate's C-verified striped `forward_filter_score`
    // (which is multihit-local, nats). Anchors the whole engine (builder +
    // recurrence + xE reduction + rescaling + score) against the C reference.
    #[test]
    fn forward_score_matches_forward_filter_score() {
        // Sweep small models, the m<100 / m>=100 DD-branch boundary (98..103),
        // and larger m; multiple lengths and per-(m,l) seeds so the striped DD
        // early-exit and xE lane reduction are exercised bit-for-bit vs the
        // crate's C-verified striped forward_filter_score.
        let ms = [
            2usize, 3, 7, 20, 33, 64, 96, 98, 99, 100, 101, 102, 103, 128, 200, 257,
        ];
        for &m in &ms {
            for seed in 0..3u64 {
                let p7 = synthetic_p7(m, 0x1234_5678 ^ (m as u64) << 3 ^ seed);
                let my_ff = build_forward_filter(&p7);
                let ref_ff = crate::cm_pipeline::build_forward_filter(&p7);
                for &l in &[10usize, 50, 200, 501] {
                    let dsq = rand_dsq(l, 0xABCD ^ (m as u64) << 8 ^ (l as u64) << 2 ^ seed);
                    let ox = p7_forward(&my_ff, &XFactors::multihit(l), &dsq, l);
                    let mine = forward_score(&ox, &XFactors::multihit(l));
                    let refsc = crate::cm_pipeline::forward_filter_score(&ref_ff, &dsq, l);
                    assert_eq!(
                        mine.to_bits(),
                        refsc.to_bits(),
                        "Forward score mismatch m={m} l={l} seed={seed}: mine={mine} ref={refsc}"
                    );
                }
            }
        }
    }

    // Real-model cross-check: load a filter HMM from an actual .cm and confirm
    // the Forward score matches the crate's C-verified striped forward_filter_score
    // bit-for-bit on real p7 emission/transition data (validates build_forward_filter
    // on non-synthetic profiles). Gated on INFERNOX_TEST_CM=<path to .cm>.
    #[test]
    fn forward_score_matches_on_real_cm() {
        let path = match std::env::var("INFERNOX_TEST_CM") {
            Ok(p) => p,
            Err(_) => return, // skipped in normal runs
        };
        let cm = crate::cm_file::cm_file_read(&path).expect("read cm");
        let p7 = cm.p7.as_ref().expect("cm has a p7 filter HMM");
        let m = p7.m as usize;
        let my_ff = build_forward_filter(p7);
        let ref_ff = crate::cm_pipeline::build_forward_filter(p7);
        for &l in &[20usize, 75, 150, 400] {
            for seed in 0..4u64 {
                let dsq = rand_dsq(l, 0xF00D ^ (l as u64) << 4 ^ seed);
                let ox = p7_forward(&my_ff, &XFactors::multihit(l), &dsq, l);
                let mine = forward_score(&ox, &XFactors::multihit(l));
                let refsc = crate::cm_pipeline::forward_filter_score(&ref_ff, &dsq, l);
                assert_eq!(
                    mine.to_bits(),
                    refsc.to_bits(),
                    "real-cm Forward mismatch m={m} l={l} seed={seed}: mine={mine} ref={refsc}"
                );
            }
        }
        eprintln!("real-cm forward OK: m={m}");
    }

    // Matrix-level check: the retained M/I/D cells are per-cell independent float
    // ops, so they must equal an independent natural-order recomputation exactly.
    #[test]
    fn matrix_cells_are_selfconsistent() {
        let m = 40usize;
        let p7 = synthetic_p7(m, 0xDEAD_BEEF);
        let ff = build_forward_filter(&p7);
        let l = 120usize;
        let dsq = rand_dsq(l, 0x5151);
        let xf = XFactors::multihit(l);
        let ox = p7_forward(&ff, &xf, &dsq, l);
        // Re-derive row i M/I/D from row i-1 with the same per-cell recurrence and
        // the SAME per-row rescale factor `scale[i]`, and compare bit-for-bit.
        for i in 1..=l {
            let x = dsq[i] as usize;
            let base = x * (m + 1);
            let inv = if ox.scale[i] != 1.0 { 1.0 / ox.scale[i] } else { 1.0 };
            // xb of the PREVIOUS row drives B->M of row i (ox.xb[i-1]).
            let xb_prev = ox.xb[i - 1];
            for k in 1..=m {
                let sv = xb_prev * ff.tbm[k]
                    + ox.mmx[i - 1][k - 1] * ff.amm[k]
                    + ox.imx[i - 1][k - 1] * ff.aim[k]
                    + ox.dmx[i - 1][k - 1] * ff.adm[k];
                let mut mval = sv * ff.rfv_t[base + k];
                let mut ival = ox.mmx[i - 1][k] * ff.tmi[k] + ox.imx[i - 1][k] * ff.tii[k];
                mval *= inv;
                ival *= inv;
                assert_eq!(mval.to_bits(), ox.mmx[i][k].to_bits(), "M[{i}][{k}]");
                assert_eq!(ival.to_bits(), ox.imx[i][k].to_bits(), "I[{i}][{k}]");
            }
        }
    }
}
