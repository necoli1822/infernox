//! cm_calibrate — faithful port of Infernal 1.1.5 `cmcalibrate`.
//!
//! Fits exponential-tail E-value parameters for a covariance model by scoring
//! random "genomic" sequences with the CM in all four exp modes
//! (EXP_CM_GC/GI/LC/LI = glocal/local × CYK/Inside) and fitting the score-tail
//! to an exponential. Mirrors cmcalibrate.c (serial_master / process_search_workunit /
//! fit_histogram) and stats.c (CreateGenomicHMM / SampleGenomicSequenceFromHMM /
//! SetExpInfo).
//!
//! RNG: cmcalibrate.c:1839 uses `esl_randomness_Create(seed)` = Mersenne Twister
//! (MT19937) = `crate::easel::random::EslRandom`. (Verified: `esl_randomness_Create`
//! creates `eslRND_MERSENNE`; cmemit is the one that uses CreateFast/LCG.)
//!
//! The random-sequence CM search here is the non-HMM, whole-sequence QDB scan
//! (cmcalibrate.c:process_search_workunit → FastCYKScan / FastIInsideScan). The
//! byte-parity-verified generic scanner lives in `cm_nohmm` but is (a) private and
//! (b) global-only; cmcalibrate needs both global and local. So the DP scanner is
//! re-ported here (float CYK / integer Inside, global + local-begin), faithfully
//! mirroring `cm_nohmm::generic_scan` (cm_dpsearch.c FastCYKScan/FastIInsideScan)
//! plus the CMH_LOCAL_BEGIN root block (cm_dpsearch.c:587-621). The global path is
//! cross-checked against the public `cm_nohmm::final_stage_inside_opt`.

use crate::cm::{abc_fcount_frac, abc_iavg_score, CM, ALPHABET_SIZE, ALPHABET_SIZE_P};
use crate::constants::{B_ST, BEGL_S, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST};
use crate::cp9::{ilogsum, score_correction_null3, INFTY, INTSCALE};
use crate::evalue::ExpParams;
use crate::easel::exponential::esl_exp_FitComplete;
use crate::easel::histogram::EslHistogram;
use crate::easel::random::EslRandom;

const IMPOSSIBLE: f32 = -1.0e36;
const EXPTAIL_CHUNKLEN: i32 = 10000; // cmcalibrate.c:52

// Exp modes, infernal.h. EXP_CM_GC=0 glocal CYK, GI=1 glocal Inside,
// LC=2 local CYK, LI=3 local Inside.
pub const EXP_CM_GC: usize = 0;
pub const EXP_CM_GI: usize = 1;
pub const EXP_CM_LC: usize = 2;
pub const EXP_CM_LI: usize = 3;
pub const EXP_NMODES: usize = 4;

#[inline]
fn exp_mode_is_local(m: usize) -> bool {
    m == EXP_CM_LC || m == EXP_CM_LI
}
#[inline]
fn exp_mode_is_inside(m: usize) -> bool {
    m == EXP_CM_GI || m == EXP_CM_LI
}

#[inline]
fn not_impossible(x: f32) -> bool {
    x > IMPOSSIBLE + 1.0
}

// ===========================================================================
// esl_vectorops: DNorm (esl_vectorops.c) — normalize a double vector to sum 1.
// ===========================================================================
fn esl_vec_dnorm(v: &mut [f64]) {
    let sum: f64 = v.iter().sum();
    if sum != 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    } else {
        let n = v.len() as f64;
        for x in v.iter_mut() {
            *x = 1.0 / n;
        }
    }
}

// ===========================================================================
// Genomic HMM — stats.c:CreateGenomicHMM (stats.c:579). Ported VERBATIM.
// A fully-connected 5-state HMM trained by EM on 30 Mb of real genomic sequence.
// ===========================================================================
pub struct GenomicHmm {
    pub nstates: usize,
    pub s_a: Vec<f64>,       // start probs [nstates]
    pub t_aa: Vec<Vec<f64>>, // transition probs [nstates][nstates]
    pub e_aa: Vec<Vec<f64>>, // emission probs [nstates][K]
}

/// C `CreateGenomicHMM` (stats.c:579). `k` = abc->K (4 for RNA/DNA).
pub fn create_genomic_hmm(k: usize) -> GenomicHmm {
    let nstates = 5;

    // start probabilities (stats.c:593)
    let mut s_a = vec![
        0.157377049180328,
        0.39344262295082,
        0.265573770491803,
        0.00327868852459016,
        0.180327868852459,
    ];
    esl_vec_dnorm(&mut s_a);

    // transition probabilities (stats.c:605)
    let mut t_aa = vec![
        vec![
            0.999483637183643,
            0.000317942006440604,
            0.000185401071732768,
            2.60394763669618e-07,
            1.27593434198113e-05,
        ],
        vec![
            9.76333640771184e-05,
            0.99980020511745,
            9.191359010352e-05,
            7.94413051888677e-08,
            1.01684870641751e-05,
        ],
        vec![
            1.3223694798182e-07,
            0.000155642887774602,
            0.999700615549769,
            9.15079680034191e-05,
            5.21013575048369e-05,
        ],
        vec![
            0.994252873563218,
            0.0014367816091954,
            0.0014367816091954,
            0.0014367816091954,
            0.0014367816091954,
        ],
        vec![
            8.32138798088677e-06,
            2.16356087503056e-05,
            6.42411152124459e-05,
            1.66427759617735e-07,
            0.999905635460297,
        ],
    ];
    for row in t_aa.iter_mut() {
        esl_vec_dnorm(row);
    }

    // emission probabilities (stats.c:645). K columns (K==4).
    let mut e_full = [
        [0.370906566523225, 0.129213995153577, 0.130511270043053, 0.369368168280145],
        [0.305194882571888, 0.194580936415687, 0.192343972160245, 0.307880208852179],
        [0.238484980800698, 0.261262845707113, 0.261810301531792, 0.238441871960397],
        [0.699280575539568, 0.00143884892086331, 0.00143884892086331, 0.297841726618705],
        [0.169064007664923, 0.331718611320207, 0.33045427183482, 0.16876310918005],
    ];
    let mut e_aa: Vec<Vec<f64>> = Vec::with_capacity(nstates);
    for row in e_full.iter_mut() {
        let mut v: Vec<f64> = row[..k].to_vec();
        esl_vec_dnorm(&mut v);
        e_aa.push(v);
    }

    GenomicHmm { nstates, s_a, t_aa, e_aa }
}

// ===========================================================================
// esl_rnd_DChoose (esl_random.c:830) — random choice from normalized discrete
// distribution `p[0..n-1]`, using esl_random(r) roll and cumulative sum/norm.
// ===========================================================================
#[inline]
pub fn esl_rnd_dchoose(r: &mut EslRandom, p: &[f64]) -> usize {
    let roll = r.random(); // esl_random(r): [0,1)
    let mut norm = 0.0f64;
    for &pi in p {
        norm += pi;
    }
    let mut sum = 0.0f64;
    for (i, &pi) in p.iter().enumerate() {
        sum += pi;
        if roll < (sum / norm) {
            return i;
        }
    }
    // C: "unreached code was reached" — return last index defensively.
    p.len() - 1
}

// ===========================================================================
// SampleGenomicSequenceFromHMM (stats.c:698). Returns dsq of length L+2 with
// dsq[0] = dsq[L+1] = 255 (eslDSQ_SENTINEL), residues at 1..=L.
// ===========================================================================
pub fn sample_genomic_sequence_from_hmm(r: &mut EslRandom, ghmm: &GenomicHmm, l: i32) -> Vec<u8> {
    let l = l as usize;
    let mut dsq = vec![0u8; l + 2];
    dsq[0] = 255;
    dsq[l + 1] = 255;

    // pick initial state (stats.c:709)
    let mut si = esl_rnd_dchoose(r, &ghmm.s_a);
    for x in 1..=l {
        dsq[x] = esl_rnd_dchoose(r, &ghmm.e_aa[si]) as u8; // emit residue
        si = esl_rnd_dchoose(r, &ghmm.t_aa[si]); // make transition
    }
    dsq
}

// ===========================================================================
// get_random_dsq (cmcalibrate.c:2037) — --random path: iid from dnull. Only
// used with --random/--gc; the default path uses the genomic HMM. `distro` is
// the double null distribution (set_dnull). esl_rsq_xIID (esl_randomseq.c).
// ===========================================================================
pub fn set_dnull(cm: &CM) -> Vec<f64> {
    // cmcalibrate.c:2076 set_dnull: double copy of cm->null, renormalized.
    let mut dnull: Vec<f64> = (0..ALPHABET_SIZE).map(|i| cm.null[i] as f64).collect();
    esl_vec_dnorm(&mut dnull);
    dnull
}

pub fn get_random_dsq_iid(r: &mut EslRandom, distro: &[f64], l: i32) -> Vec<u8> {
    // esl_rsq_xIID: dsq[x] = esl_rnd_DChoose(r, distro, K)
    let l = l as usize;
    let mut dsq = vec![0u8; l + 2];
    dsq[0] = 255;
    dsq[l + 1] = 255;
    for x in 1..=l {
        dsq[x] = esl_rnd_dchoose(r, distro) as u8;
    }
    dsq
}

// ===========================================================================
// DP scanner: unified CYK (f32, ESL_MAX) / Inside (i32, ILogsum), global +
// local-begin. Faithful re-port of cm_nohmm::generic_scan (verified global)
// plus the CMH_LOCAL_BEGIN root block (cm_dpsearch.c FastCYKScan:587-621 /
// FastIInsideScan). Returns greedily-resolved, overlap-pruned hits.
// ===========================================================================

// C Prob2Score (cm.c:4220). (public in cm_nohmm; re-derived here to build the
// integer LOCAL scores, which cm_configure_scores does not build.)
#[inline]
fn prob2score(p: f32, null: f32) -> i32 {
    if p == 0.0 {
        -INFTY
    } else {
        (0.5 + INTSCALE * (p as f64 / null as f64).log2()).floor() as i32
    }
}
#[inline]
fn scorify(sc: i32) -> f32 {
    if sc == -INFTY {
        IMPOSSIBLE
    } else {
        (sc as f64 / INTSCALE) as f32
    }
}

// Emit modes (C Emitmode()).
const EMITNONE: i32 = 0;
const EMITLEFT: i32 = 1;
const EMITRIGHT: i32 = 2;
const EMITPAIR: i32 = 3;
#[inline]
fn emitmode(stt: i32) -> i32 {
    match stt {
        x if x == ML_ST || x == IL_ST => EMITLEFT,
        x if x == MR_ST || x == IR_ST => EMITRIGHT,
        x if x == MP_ST => EMITPAIR,
        _ => EMITNONE,
    }
}
#[inline]
fn state_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST => 2,
        x if x == ML_ST || x == MR_ST || x == IL_ST || x == IR_ST => 1,
        _ => 0,
    }
}
#[inline]
fn state_right_delta(stt: i32) -> i32 {
    match stt {
        x if x == MP_ST || x == MR_ST || x == IR_ST => 1,
        _ => 0,
    }
}

trait Sr: Copy {
    fn zero() -> Self;
    fn from_zero() -> Self;
    fn comb(a: Self, b: Self) -> Self;
    fn vmax(a: Self, b: Self) -> Self; // ESL_MAX (Viterbi max) for BOTH semirings
    fn addv(a: Self, b: Self) -> Self;
    fn scbits(a: Self) -> f32;
    fn valid(a: Self) -> bool; // NOT_IMPOSSIBLE / > -INFTY
}
impl Sr for f32 {
    #[inline]
    fn zero() -> Self {
        IMPOSSIBLE
    }
    #[inline]
    fn from_zero() -> Self {
        0.0
    }
    #[inline]
    fn comb(a: Self, b: Self) -> Self {
        if a >= b {
            a
        } else {
            b
        }
    }
    #[inline]
    fn vmax(a: Self, b: Self) -> Self {
        if a >= b {
            a
        } else {
            b
        }
    }
    #[inline]
    fn addv(a: Self, b: Self) -> Self {
        a + b
    }
    #[inline]
    fn scbits(a: Self) -> f32 {
        a
    }
    #[inline]
    fn valid(a: Self) -> bool {
        not_impossible(a)
    }
}
impl Sr for i32 {
    #[inline]
    fn zero() -> Self {
        -INFTY
    }
    #[inline]
    fn from_zero() -> Self {
        0
    }
    #[inline]
    fn comb(a: Self, b: Self) -> Self {
        ilogsum(a, b)
    }
    #[inline]
    fn vmax(a: Self, b: Self) -> Self {
        if a >= b {
            a
        } else {
            b
        }
    }
    #[inline]
    fn addv(a: Self, b: Self) -> Self {
        a.wrapping_add(b)
    }
    #[inline]
    fn scbits(a: Self) -> f32 {
        scorify(a)
    }
    #[inline]
    fn valid(a: Self) -> bool {
        a > -INFTY
    }
}

/// Reduce `init` + child `terms[0..cnum]` in C's EXACT hand-unrolled ILogsum
/// order (cm_dpsearch.c FastIInsideScan switch(cnum), lines 1352-1475 /
/// 1498-1580; identical in the IL/IR and general-emitter blocks). Integer
/// ILogsum is not associative (lookup-table rounding), so the order is
/// load-bearing for Inside byte-parity. Order is harmless (order-independent)
/// for the CYK/MAX semiring. `init` = init_scAA[v][dp_y]; `terms[k]` =
/// alpha[jp_y][y+k][dp_y] + tsc_v[k].
#[inline]
fn reduce_c_order<S: Sr>(cnum: usize, terms: &[S], init: S) -> S {
    match cnum {
        2 => {
            // [1, init, 0]
            let mut s = S::comb(terms[1], init);
            s = S::comb(s, terms[0]);
            s
        }
        3 => {
            // [2, 1, init, 0]
            let mut s = S::comb(terms[2], terms[1]);
            s = S::comb(s, init);
            s = S::comb(s, terms[0]);
            s
        }
        4 => {
            // [3, 2, init, 1, 0]
            let mut s = S::comb(terms[3], terms[2]);
            s = S::comb(s, init);
            s = S::comb(s, terms[1]);
            s = S::comb(s, terms[0]);
            s
        }
        5 => {
            // [4, 3, init, 1, 2, 0]
            let mut s = S::comb(terms[4], terms[3]);
            s = S::comb(s, init);
            s = S::comb(s, terms[1]);
            s = S::comb(s, terms[2]);
            s = S::comb(s, terms[0]);
            s
        }
        6 => {
            // [5, init, 4, 3, 2, 1, 0]
            let mut s = S::comb(terms[5], init);
            s = S::comb(s, terms[4]);
            s = S::comb(s, terms[3]);
            s = S::comb(s, terms[2]);
            s = S::comb(s, terms[1]);
            s = S::comb(s, terms[0]);
            s
        }
        1 => S::comb(init, terms[0]),
        _ => {
            // cnum 0 or >6: no C unrolled case reaches these emitter branches;
            // natural order fallback.
            let mut s = init;
            for &t in &terms[..cnum] {
                s = S::comb(s, t);
            }
            s
        }
    }
}

struct Tables<'a, S: Sr> {
    tsc: &'a [Vec<S>],
    oesc: &'a [Vec<S>],
    endsc: Vec<S>,
    beginsc: Vec<S>, // per-state local-begin score (IMPOSSIBLE/-INFTY in global)
    el_self: S,
}

#[derive(Clone, Copy)]
struct RawHit {
    i: i32,
    j: i32,
    score: f32,
    bias: f32,
}

#[allow(clippy::too_many_arguments)]
fn generic_scan<S: Sr>(
    cm: &CM,
    tab: &Tables<S>,
    dsq: &[u8],
    i0: i32,
    j0: i32,
    cutoff: f32,
    do_null3: bool,
    do_local: bool,
    qdb_loose: bool,
) -> Vec<RawHit> {
    let m = cm.m as usize;
    let kp = ALPHABET_SIZE_P;
    let smx_w = cm.w;
    let l = j0 - i0 + 1;
    let mut w = smx_w;
    if w > l {
        w = l;
    }
    let w = w as usize;

    // QDB band vectors (beta1=beta2 in calibration, so LOOSE==TIGHT here).
    let (dmin, dmax): (&[i32], &[i32]) = if qdb_loose {
        (&cm.dmin2, &cm.dmax2)
    } else {
        (&cm.dmin1, &cm.dmax1)
    };

    let mut dn_v = vec![0i32; m];
    let mut dxcap_v = vec![0i32; m];
    for v in 0..m {
        let base = if cm.sttype[v] as i32 == MP_ST { 2 } else { 1 };
        let mut dn = dmin[v].max(base);
        if dn > smx_w {
            dn = smx_w;
        }
        dn_v[v] = dn;
        dxcap_v[v] = dmax[v].min(smx_w);
    }

    let mut begl_idx = vec![usize::MAX; m];
    let mut nbegl = 0usize;
    for v in 0..m {
        if cm.stid[v] as i32 == BEGL_S {
            begl_idx[v] = nbegl;
            nbegl += 1;
        }
    }

    let mut el_sca = vec![S::from_zero(); w + 1];
    for d in 1..=w {
        el_sca[d] = S::addv(el_sca[d - 1], tab.el_self);
    }
    let mut init_sc = vec![vec![S::zero(); w + 1]; m];
    for v in 0..m {
        let es = tab.endsc[v];
        if S::valid(es) {
            for d in 0..=w {
                init_sc[v][d] = S::addv(el_sca[d], es);
            }
        }
    }

    let mut alpha = vec![vec![vec![S::zero(); w + 1]; m]; 2];
    let mut alpha_begl = vec![vec![vec![S::zero(); w + 1]; nbegl]; w + 1];

    // ---- d=0 base cases ----
    for v in (0..m).rev() {
        if cm.stid[v] as i32 != BEGL_S {
            let stt = cm.sttype[v] as i32;
            if stt == E_ST {
                alpha[0][v][0] = S::from_zero();
                alpha[1][v][0] = S::from_zero();
            } else if stt == S_ST || stt == D_ST {
                // C cm_scan_mx_InitializeIntegers/Floats (cm_mx.c:6112-6118): the
                // d=0 base case uses ESL_MAX (Viterbi max), NOT ILogsum, even for
                // the integer Inside scan. Using comb (ILogsum) here over-counts
                // the multiple finite local-end (iendsc) child terms present in
                // local config, diverging ECMLI from C. (Glocal: only isolated
                // finite d=0 children, so MAX==ILogsum and ECMGI is unaffected.)
                let y = cm.cfirst[v] as usize;
                let mut a0 = tab.endsc[v];
                for yo in 0..cm.cnum[v] as usize {
                    a0 = S::vmax(a0, S::addv(alpha[0][y + yo][0], tab.tsc[v][yo]));
                }
                a0 = S::vmax(a0, S::zero());
                alpha[0][v][0] = a0;
                alpha[1][v][0] = a0;
            } else if stt == B_ST {
                let wl = cm.cfirst[v] as usize;
                let y = cm.cnum[v] as usize;
                let a0 = S::addv(alpha_begl[0][begl_idx[wl]][0], alpha[0][y][0]);
                alpha[0][v][0] = a0;
                alpha[1][v][0] = a0;
            } else {
                alpha[1][v][0] = alpha[0][v][0];
            }
        } else {
            // BEGL_S d=0 base case: ESL_MAX (Viterbi max), matching C
            // cm_scan_mx_InitializeIntegers (cm_mx.c:6128-6131), NOT ILogsum.
            let bi = begl_idx[v];
            let y = cm.cfirst[v] as usize;
            let mut a0 = tab.endsc[v];
            for yo in 0..cm.cnum[v] as usize {
                a0 = S::vmax(a0, S::addv(alpha[0][y + yo][0], tab.tsc[v][yo]));
            }
            a0 = S::vmax(a0, S::zero());
            for j in 0..=w {
                alpha_begl[j][bi][0] = a0;
            }
        }
    }

    let mut act: Vec<[f64; 4]> = if do_null3 {
        vec![[0.0; 4]; w + 1]
    } else {
        Vec::new()
    };

    let mut jp_wa = vec![0usize; w + 1];
    let mut sc_v = vec![S::zero(); w + 1];
    let mut bestsc = vec![IMPOSSIBLE; w + 1];
    let mut bestr = vec![-1i32; w + 1];
    let mut tmp_hits: Vec<RawHit> = Vec::new();

    for j in i0..=j0 {
        let jp_g = (j - i0 + 1) as usize;
        let cur = (j & 1) as usize;
        let prv = ((j - 1) & 1) as usize;
        let jrow = jp_g.min(w);
        for d in 0..=w {
            jp_wa[d] = ((j - d as i32).rem_euclid(w as i32 + 1)) as usize;
        }
        if do_null3 {
            let src = (jp_g - 1) % (w + 1);
            let dst = jp_g % (w + 1);
            act[dst] = act[src];
            let dj = dsq[j as usize];
            if (dj as usize) < 4 {
                act[dst][dj as usize] += 1.0;
            }
        }

        for v in (1..m).rev() {
            let stt = cm.sttype[v] as i32;
            if stt == E_ST {
                continue;
            }
            let sd = state_delta(stt);
            let em = emitmode(stt);
            let jp_v = if cm.stid[v] as i32 == BEGL_S {
                (j.rem_euclid(w as i32 + 1)) as usize
            } else {
                cur
            };
            let jp_y = if state_right_delta(stt) > 0 { prv } else { cur };
            let dn = dn_v[v];
            let dx = (jrow as i32).min(dxcap_v[v]);
            let esc_v: &[S] = &tab.oesc[v];
            let esc_j = if stt == IR_ST || stt == MR_ST {
                esc_v[dsq[j as usize] as usize]
            } else {
                S::zero()
            };

            if stt == B_ST {
                let wl = cm.cfirst[v] as usize;
                let y = cm.cnum[v] as usize;
                let bi = begl_idx[wl];
                for d in dn..=dx {
                    let dn_y = dmin[y].min(smx_w);
                    let dx_y = dmax[y].min(smx_w);
                    let dn_w = dmin[wl].min(smx_w);
                    let dx_w = dmax[wl].min(smx_w);
                    let kmin = 0.max(dn_y.max(d - dx_w));
                    let kmax = dx_y.min(d - dn_w);
                    let mut sc = init_sc[v][(d - sd) as usize];
                    let mut kk = kmin;
                    while kk <= kmax {
                        let left = alpha_begl[jp_wa[kk as usize]][bi][(d - kk) as usize];
                        let right = alpha[jp_y][y][kk as usize];
                        sc = S::comb(sc, S::addv(left, right));
                        kk += 1;
                    }
                    alpha[jp_v][v][d as usize] = sc;
                }
            } else if cm.stid[v] as i32 == BEGL_S {
                let y = cm.cfirst[v] as usize;
                let bi = begl_idx[v];
                for d in dn..=dx {
                    let mut sc = init_sc[v][(d - sd) as usize];
                    for yo in 0..cm.cnum[v] as usize {
                        sc = S::comb(
                            sc,
                            S::addv(alpha[jp_y][y + yo][(d - sd) as usize], tab.tsc[v][yo]),
                        );
                    }
                    alpha_begl[jp_v][bi][d as usize] = sc;
                }
            } else if stt == IL_ST || stt == IR_ST {
                let y = cm.cfirst[v] as usize;
                let tsc_v = &tab.tsc[v];
                let cnum = cm.cnum[v] as usize;
                let mut i = j - dn + 1;
                let mut dp_y = dn - sd;
                for d in dn..=dx {
                    let mut terms = [S::zero(); 6];
                    for yo in 0..cnum {
                        terms[yo] = S::addv(alpha[jp_y][y + yo][dp_y as usize], tsc_v[yo]);
                    }
                    let mut sc = reduce_c_order::<S>(cnum, &terms, init_sc[v][dp_y as usize]);
                    match em {
                        EMITLEFT => {
                            sc = S::addv(sc, esc_v[dsq[i as usize] as usize]);
                            i -= 1;
                        }
                        EMITRIGHT => {
                            sc = S::addv(sc, esc_j);
                        }
                        _ => {}
                    }
                    alpha[jp_v][v][d as usize] = sc;
                    dp_y += 1;
                }
            } else {
                let y = cm.cfirst[v] as usize;
                let tsc_v = &tab.tsc[v];
                let cnum = cm.cnum[v] as usize;
                let mut dp_y = dn - sd;
                for d in dn..=dx {
                    let mut terms = [S::zero(); 6];
                    for yo in 0..cnum {
                        terms[yo] = S::addv(alpha[jp_y][y + yo][dp_y as usize], tsc_v[yo]);
                    }
                    sc_v[d as usize] = reduce_c_order::<S>(cnum, &terms, init_sc[v][dp_y as usize]);
                    dp_y += 1;
                }
                match em {
                    EMITLEFT => {
                        let mut i = j - dn + 1;
                        for d in dn..=dx {
                            alpha[jp_v][v][d as usize] =
                                S::addv(sc_v[d as usize], esc_v[dsq[i as usize] as usize]);
                            i -= 1;
                        }
                    }
                    EMITNONE => {
                        for d in dn..=dx {
                            alpha[jp_v][v][d as usize] = sc_v[d as usize];
                        }
                    }
                    EMITRIGHT => {
                        for d in dn..=dx {
                            alpha[jp_v][v][d as usize] = S::addv(sc_v[d as usize], esc_j);
                        }
                    }
                    EMITPAIR => {
                        let mut i = j - dn + 1;
                        for d in dn..=dx {
                            let idx = dsq[i as usize] as usize * kp + dsq[j as usize] as usize;
                            alpha[jp_v][v][d as usize] = S::addv(sc_v[d as usize], esc_v[idx]);
                            i -= 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        // ---- ROOT_S (v=0): plain root transitions, then local begins ----
        let dn0 = dn_v[0];
        let dx0 = (jrow as i32).min(dxcap_v[0]);
        let tsc0 = &tab.tsc[0];
        let y0 = cm.cfirst[0] as usize;
        for d in dn0..=dx0 {
            bestr[d as usize] = 0;
            let mut a = S::comb(S::zero(), S::addv(alpha[cur][y0][d as usize], tsc0[0]));
            for yo in 1..cm.cnum[0] as usize {
                a = S::comb(a, S::addv(alpha[cur][y0 + yo][d as usize], tsc0[yo]));
            }
            alpha[cur][0][d as usize] = a;
        }
        // CMH_LOCAL_BEGIN block (cm_dpsearch.c:587-621).
        if do_local {
            for y in 1..m {
                if !S::valid(tab.beginsc[y]) {
                    continue;
                }
                let dn = dn_v[0].max(dn_v[y]);
                let dx_here = dx0.min((jrow as i32).min(dxcap_v[y]));
                if cm.stid[y] as i32 == BEGL_S {
                    let jp_y = (j.rem_euclid(w as i32 + 1)) as usize;
                    let bi = begl_idx[y];
                    for d in dn..=dx_here {
                        let cand = S::addv(alpha_begl[jp_y][bi][d as usize], tab.beginsc[y]);
                        if S::scbits(alpha[cur][0][d as usize]) < S::scbits(cand) {
                            alpha[cur][0][d as usize] = cand;
                            bestr[d as usize] = y as i32;
                        }
                    }
                } else {
                    for d in dn..=dx_here {
                        let cand = S::addv(alpha[cur][y][d as usize], tab.beginsc[y]);
                        if S::scbits(alpha[cur][0][d as usize]) < S::scbits(cand) {
                            alpha[cur][0][d as usize] = cand;
                            bestr[d as usize] = y as i32;
                        }
                    }
                }
            }
        }
        for d in dn0..=dx0 {
            bestsc[d as usize] = S::scbits(alpha[cur][0][d as usize]);
        }

        report_hits_greedily(
            cm,
            j,
            dn0,
            dx0,
            &bestsc,
            &bestr,
            w,
            if do_null3 { Some(&act) } else { None },
            i0,
            cutoff,
            &mut tmp_hits,
        );
    }

    // ---- greedy overlap removal ----
    let raw: Vec<(i32, i32, f32, f32)> =
        tmp_hits.iter().map(|h| (h.i, h.j, h.score, h.bias)).collect();
    let kept = crate::cp9::remove_overlaps_greedy(raw);
    kept.into_iter()
        .map(|(i, j, s, b)| RawHit { i, j, score: s, bias: b })
        .collect()
}

// C ReportHitsGreedily (cm_mx.c), std (non-trunc). Mirrors cm_nohmm.
#[allow(clippy::too_many_arguments)]
fn report_hits_greedily(
    cm: &CM,
    j: i32,
    dmin: i32,
    dmax: i32,
    bestsc: &[f32],
    bestr: &[i32],
    w: usize,
    act: Option<&Vec<[f64; 4]>>,
    i0: i32,
    cutoff: f32,
    out: &mut Vec<RawHit>,
) {
    if dmin > dmax {
        return;
    }
    let mut max_reported = IMPOSSIBLE;
    let dlo = dmin.max(1);
    for d in dlo..=dmax {
        let i = j - d + 1;
        let mut hit_sc = bestsc[d as usize];
        let mut bias = 0.0f32;
        if hit_sc > max_reported && hit_sc >= cutoff && not_impossible(hit_sc) {
            let mut do_report = true;
            if let Some(act) = act {
                let ip = (i - i0 + 1) as usize;
                let jp = (j - i0 + 1) as usize;
                let mut comp = [0.0f32; 4];
                for a in 0..4 {
                    comp[a] = (act[jp % (w + 1)][a] - act[(ip - 1) % (w + 1)][a]) as f32;
                }
                esl_vec_fnorm(&mut comp);
                let corr = score_correction_null3(&cm.null, &comp, d, cm.n3_omega as f32);
                hit_sc -= corr;
                bias = corr;
                do_report = hit_sc > max_reported && hit_sc >= cutoff;
            }
            if do_report {
                out.push(RawHit { i, j, score: hit_sc, bias });
                max_reported = hit_sc;
            }
        }
    }
}

#[inline]
fn esl_vec_fnorm(v: &mut [f32]) {
    let sum: f32 = v.iter().sum();
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    } else {
        let n = v.len() as f32;
        for x in v.iter_mut() {
            *x = 1.0 / n;
        }
    }
}

// ===========================================================================
// Integer score builder for LOCAL config. cm_configure_scores (cp9)
// localizes cm.t + builds FLOAT scores only; FastIInsideScan needs INTEGER
// scores. Mirrors cm_nohmm::build_integer_scores (CMLogoddsify +
// ICalcOptimizedEmitScores) on the (localized) probabilities.
// ===========================================================================
fn build_integer_scores(cm: &mut CM) {
    let m = cm.m as usize;
    let k = ALPHABET_SIZE;
    let kp = ALPHABET_SIZE_P;
    cm.itsc = vec![Vec::new(); m];
    cm.ioesc = vec![Vec::new(); m];
    cm.ibeginsc = vec![-INFTY; m];
    cm.iendsc = vec![-INFTY; m];
    for v in 0..m {
        let stt = cm.sttype[v] as i32;
        if stt != B_ST && stt != E_ST {
            let cnum = cm.cnum[v] as usize;
            let mut itsc = vec![0i32; cnum];
            for x in 0..cnum {
                itsc[x] = prob2score(cm.t[v][x], 1.0);
            }
            cm.itsc[v] = itsc;
        }
        if stt == MP_ST {
            let mut iesc = vec![0i32; k * k];
            for a in 0..k {
                for b in 0..k {
                    iesc[a * k + b] = prob2score(cm.e[v][a * k + b], cm.null[a] * cm.null[b]);
                }
            }
            let mut ioesc = vec![-INFTY; kp * kp];
            for a in 0..k {
                for b in 0..k {
                    ioesc[a * kp + b] = iesc[a * k + b];
                }
            }
            for a in (k + 1)..(kp - 1) {
                for b in 0..k {
                    let mut sc = 0.0f32;
                    for l in 0..k {
                        sc += iesc[l * k + b] as f32 * abc_fcount_frac(a, l);
                    }
                    ioesc[a * kp + b] = sc as i32;
                }
            }
            for a in 0..k {
                for b in (k + 1)..(kp - 1) {
                    let mut sc = 0.0f32;
                    for rr in 0..k {
                        sc += iesc[a * k + rr] as f32 * abc_fcount_frac(b, rr);
                    }
                    ioesc[a * kp + b] = sc as i32;
                }
            }
            for a in (k + 1)..(kp - 1) {
                for b in (k + 1)..(kp - 1) {
                    let mut sc = 0.0f32;
                    for l in 0..k {
                        for rr in 0..k {
                            sc += iesc[l * k + rr] as f32
                                * abc_fcount_frac(a, l)
                                * abc_fcount_frac(b, rr);
                        }
                    }
                    ioesc[a * kp + b] = sc as i32;
                }
            }
            cm.ioesc[v] = ioesc;
        } else if stt == ML_ST || stt == MR_ST || stt == IL_ST || stt == IR_ST {
            let mut iesc = vec![0i32; k];
            for a in 0..k {
                iesc[a] = prob2score(cm.e[v][a], cm.null[a]);
            }
            let mut ioesc = vec![-INFTY; kp];
            for a in 0..k {
                ioesc[a] = iesc[a];
            }
            for a in (k + 1)..(kp - 1) {
                ioesc[a] = abc_iavg_score(a, &iesc);
            }
            cm.ioesc[v] = ioesc;
        }
        cm.ibeginsc[v] = prob2score(cm.begin[v], 1.0);
        cm.iendsc[v] = prob2score(cm.end[v], 1.0);
    }
    cm.iel_selfsc = prob2score(2f32.powf(cm.el_selfsc), 1.0);
}

fn float_tables(cm: &CM) -> Tables<'_, f32> {
    Tables {
        tsc: &cm.tsc,
        oesc: &cm.oesc,
        endsc: cm.endsc.clone(),
        beginsc: cm.beginsc.clone(),
        el_self: cm.el_selfsc,
    }
}
fn int_tables(cm: &CM) -> Tables<'_, i32> {
    Tables {
        tsc: &cm.itsc,
        oesc: &cm.ioesc,
        endsc: cm.iendsc.clone(),
        beginsc: cm.ibeginsc.clone(),
        el_self: cm.iel_selfsc,
    }
}

/// Score every random sequence with the CM in one exp mode; collect all hit
/// scores (cmcalibrate.c serial_loop → process_search_workunit; cutoff=-inf so
/// all greedily-resolved hits are collected). `seqs` are digitized (len L+2).
fn collect_scores(
    cm: &CM,
    seqs: &[Vec<u8>],
    l: i32,
    inside: bool,
    do_local: bool,
    do_null3: bool,
) -> Vec<f32> {
    let cutoff = f32::NEG_INFINITY;
    let mut out: Vec<f32> = Vec::new();
    for dsq in seqs {
        let hits = if inside {
            let tab = int_tables(cm);
            generic_scan::<i32>(cm, &tab, dsq, 1, l, cutoff, do_null3, do_local, true)
        } else {
            let tab = float_tables(cm);
            generic_scan::<f32>(cm, &tab, dsq, 1, l, cutoff, do_null3, do_local, true)
        };
        for h in hits {
            out.push(h.score);
        }
    }
    out
}

// ===========================================================================
// fit_histogram (cmcalibrate.c:1930)
// ===========================================================================
pub struct FitResult {
    pub mu_orig: f64,
    pub lambda: f64,
    pub nrandhits: i32,
    pub tailp: f32,
}

/// C fit_histogram (cmcalibrate.c:1930). Fills CreateFull(-100,100,.1), takes the
/// top tailp mass (tailp from --gtailn/--ltailn hits-per-Mb, or --tailp), and
/// fits an exponential tail (esl_exp_FitComplete).
pub fn fit_histogram(
    scores: &[f32],
    exp_mode: usize,
    n_seqs: i32,
    l: i32,
    gtailn: i32,
    ltailn: i32,
    tailp_opt: Option<f32>,
) -> Result<FitResult, String> {
    let mut h = EslHistogram::create_full(-100.0, 100.0, 0.1);
    for &s in scores {
        h.add(s as f64)
            .map_err(|e| format!("histogram Add failed: {e:?}"))?;
    }
    let hn = h.n; // total number of hits in histogram

    let tailp: f32 = if let Some(tp) = tailp_opt {
        tp
    } else {
        let mb = (n_seqs as f32 * l as f32) / 1_000_000.0;
        let nhits_to_fit = if exp_mode_is_local(exp_mode) {
            ltailn as f32 * mb
        } else {
            gtailn as f32 * mb
        };
        let tp = nhits_to_fit / hn as f32;
        if tp > 1.0 {
            return Err(format!(
                "tailn too large: only {:.3} hits/Mb in histogram; lower <n> or use --tailp",
                hn as f32 / mb
            ));
        }
        tp
    };

    let (xv, nn, _z) = h
        .get_tail_by_mass(tailp as f64)
        .map_err(|e| format!("GetTailByMass failed: {e:?}"))?;
    if nn <= 1 {
        return Err(format!(
            "too few points in tail fit ({:.6} fraction). Increase -L.",
            tailp
        ));
    }
    let (mu, lambda) =
        esl_exp_FitComplete(xv, nn).map_err(|e| format!("FitComplete failed: {e:?}"))?;
    if lambda.is_nan() {
        return Err("exp tail fit lambda is NaN; increase -L".to_string());
    }
    if lambda.is_infinite() {
        return Err("exp tail fit lambda is inf; increase -L".to_string());
    }
    Ok(FitResult {
        mu_orig: mu,
        lambda,
        nrandhits: hn as i32,
        tailp,
    })
}

// ===========================================================================
// Calibrate driver (cmcalibrate.c serial_master, single-CM, serial).
// ===========================================================================
#[derive(Clone)]
pub struct CalibrateConfig {
    pub seed: u32,
    pub l_mb: f64,   // -L
    pub beta: f64,   // --beta (QDB tail loss)
    pub nonbanded: bool,
    pub gtailn: i32, // --gtailn (default 250)
    pub ltailn: i32, // --ltailn (default 750)
    pub tailp: Option<f32>, // --tailp override
    pub do_null3: bool,     // !--nonull3
}
impl Default for CalibrateConfig {
    fn default() -> Self {
        CalibrateConfig {
            seed: 181,
            l_mb: 1.6,
            beta: 1e-15,
            nonbanded: false,
            gtailn: 250,
            ltailn: 750,
            tailp: None,
            do_null3: true,
        }
    }
}

pub struct CalibrateResult {
    pub exp: [ExpParams; EXP_NMODES], // indexed [GC, GI, LC, LI]
    pub dbsize: f64,
    pub nhits: [i32; EXP_NMODES],
    pub mu_extrap: [f64; EXP_NMODES],
    pub lambda: [f64; EXP_NMODES],
}

/// C SetExpInfo (stats.c:459): fill an ExpInfo, computing mu_extrap.
fn set_exp_info(lambda: f64, mu_orig: f64, dbsize: f64, nrandhits: i32, tailp: f32) -> ExpParams {
    let mu_extrap = mu_orig - (1.0 / tailp as f64).ln() / lambda;
    ExpParams {
        lambda,
        mu: mu_extrap,
        dbsize,
        nrandhits,
        mu_orig,
        tailp: tailp as f64,
    }
}

/// Run a full calibration for one CM. Does not modify `cm`; returns exp params.
pub fn calibrate_cm(cm: &CM, cfg: &CalibrateConfig) -> Result<CalibrateResult, String> {
    let l = EXPTAIL_CHUNKLEN;
    // N = round(L_Mb * 1e6 / L)  (cmcalibrate.c:299)
    let n_seqs = (((cfg.l_mb * 1_000_000.0) / l as f32 as f64) + 0.5) as i32;
    if n_seqs < 1 {
        return Err("-L too small: N < 1 sequence".to_string());
    }

    // ---- generate the N random genomic sequences (once, shared by all modes) ----
    // generate_sequences (cmcalibrate.c:2250): reseed r with seed (unless seed 0),
    // then sample N seqs of length L from the genomic HMM.
    let ghmm = create_genomic_hmm(ALPHABET_SIZE);
    let mut r = EslRandom::new(cfg.seed);
    if cfg.seed != 0 {
        r.init(cfg.seed);
    }
    let seqs: Vec<Vec<u8>> = (0..n_seqs)
        .map(|_| sample_genomic_sequence_from_hmm(&mut r, &ghmm, l))
        .collect();

    let dbsize = (l as f64) * (n_seqs as f64);

    // ---- GLOBAL CM: configure global scores + QDB bands ----
    let mut gcm = cm.clone();
    crate::cm_nohmm::cm_configure_scores_global(&mut gcm);
    if !cfg.nonbanded {
        crate::cm_nohmm::cm_calc_qdb_bands(&mut gcm, cfg.beta, cfg.beta, cm.w_beta)?;
    } else {
        set_full_bands(&mut gcm);
    }

    // ---- LOCAL CM: QDB on GLOBAL transitions first (cm_Configure order), then
    //      localize + build float & integer scores ----
    let mut lcm = cm.clone();
    if !cfg.nonbanded {
        crate::cm_nohmm::cm_calc_qdb_bands(&mut lcm, cfg.beta, cfg.beta, cm.w_beta)?;
    } else {
        set_full_bands(&mut lcm);
    }
    lcm.flags |= (1 << 10) | (1 << 11); // CMH_LOCAL_BEGIN | CMH_LOCAL_END
    crate::cp9::cm_configure_scores(&mut lcm); // localize + float scores
    build_integer_scores(&mut lcm); // integer scores on localized probs

    // ---- score + fit each mode ----
    let mut exp: [ExpParams; EXP_NMODES] = Default::default();
    let mut nhits = [0i32; EXP_NMODES];
    let mut mu_extrap = [0.0f64; EXP_NMODES];
    let mut lambda_out = [0.0f64; EXP_NMODES];

    for mode in 0..EXP_NMODES {
        let local = exp_mode_is_local(mode);
        let inside = exp_mode_is_inside(mode);
        let cm_use = if local { &lcm } else { &gcm };
        let scores = collect_scores(cm_use, &seqs, l, inside, local, cfg.do_null3);
        let fit = fit_histogram(&scores, mode, n_seqs, l, cfg.gtailn, cfg.ltailn, cfg.tailp)?;
        let ep = set_exp_info(fit.lambda, fit.mu_orig, dbsize, fit.nrandhits, fit.tailp);
        nhits[mode] = fit.nrandhits;
        mu_extrap[mode] = ep.mu;
        lambda_out[mode] = fit.lambda;
        exp[mode] = ep;
    }

    Ok(CalibrateResult {
        exp,
        dbsize,
        nhits,
        mu_extrap,
        lambda: lambda_out,
    })
}

/// C `forecast_time` (cmcalibrate.c:2389): estimate the wall-clock seconds to
/// calibrate one CM. Times a glocal CYK + glocal Inside search of a single
/// sequence of length L=max(2*W,200), then extrapolates to the full L*N workload
/// (2x for local+glocal) and divides by the number of workers. Returns the
/// predicted seconds (inherently machine/wall-clock dependent — the caller
/// formats it into the `hh:mm:sec` column, which normalizes out in verification,
/// exactly as C's stopwatch-derived estimate does).
///
/// NOTE: this uses a *separate* RNG (seeded from cfg.seed) mirroring C's dedicated
/// `cfg->r_est` (reseeded in forecast_time, cmcalibrate.c:2412) so it does not
/// perturb the calibration RNG used by `calibrate_cm`.
pub fn estimate_calibration_seconds(cm: &CM, cfg: &CalibrateConfig, relevant_ncpus: i32) -> f64 {
    use std::time::Instant;
    // L = ESL_MAX(2*W, 200) (cmcalibrate.c:2421).
    let lmicro = std::cmp::max(2 * cm.w, 200);
    let ghmm = create_genomic_hmm(ALPHABET_SIZE);
    let mut r_est = EslRandom::new(cfg.seed);
    if cfg.seed != 0 {
        r_est.init(cfg.seed);
    }
    let seq = sample_genomic_sequence_from_hmm(&mut r_est, &ghmm, lmicro);
    let seqs = vec![seq];

    // Configure the global CM + QDB bands exactly as calibrate_cm does.
    let mut gcm = cm.clone();
    crate::cm_nohmm::cm_configure_scores_global(&mut gcm);
    if !cfg.nonbanded {
        let _ = crate::cm_nohmm::cm_calc_qdb_bands(&mut gcm, cfg.beta, cfg.beta, cm.w_beta);
    } else {
        set_full_bands(&mut gcm);
    }

    // time glocal CYK, then glocal Inside (cmcalibrate.c:2445-2455).
    let t0 = Instant::now();
    let _ = collect_scores(&gcm, &seqs, lmicro, false, false, cfg.do_null3);
    let cyk_sec = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let _ = collect_scores(&gcm, &seqs, lmicro, true, false, cfg.do_null3);
    let ins_sec = t1.elapsed().as_secs_f64();

    let cyk_per_res = cyk_sec / lmicro as f64;
    let ins_per_res = ins_sec / lmicro as f64;
    // N = round(L_Mb * 1e6 / chunk) (cmcalibrate.c:299).
    let n_seqs = (((cfg.l_mb * 1_000_000.0) / EXPTAIL_CHUNKLEN as f64) + 0.5) as i32;
    // psec = 2*(cyk+ins)_per_res * L_full * N, then /workers (cmcalibrate.c:2465-2477).
    let mut psec = 2.0 * (cyk_per_res + ins_per_res) * (EXPTAIL_CHUNKLEN as f64) * (n_seqs as f64);
    psec /= std::cmp::max(1, relevant_ncpus) as f64;
    psec
}

// --nonbanded: full d-range (SMX_NOQDB): dmin=0, dmax=W for every state.
fn set_full_bands(cm: &mut CM) {
    let m = cm.m as usize;
    cm.dmin1 = vec![0; m];
    cm.dmax1 = vec![cm.w; m];
    cm.dmin2 = vec![0; m];
    cm.dmax2 = vec![cm.w; m];
}

/// Write the calibrated exp params into a CM (sets exp_by_mode, the four
/// exp_params_* mirrors used by the search engine, and CM_EXPTAIL_STATS).
pub fn apply_calibration(cm: &mut CM, res: &CalibrateResult) {
    cm.exp_by_mode = [
        res.exp[EXP_CM_GC].clone(),
        res.exp[EXP_CM_GI].clone(),
        res.exp[EXP_CM_LC].clone(),
        res.exp[EXP_CM_LI].clone(),
    ];
    // Mirrors used by cm_search / cm_file reader semantics.
    cm.exp_params_global_cyk = res.exp[EXP_CM_GC].clone(); // ECMGC
    cm.exp_params = res.exp[EXP_CM_GI].clone(); // ECMGI (default global Inside)
    cm.exp_params_local_cyk = res.exp[EXP_CM_LC].clone(); // ECMLC
    cm.exp_params_local = res.exp[EXP_CM_LI].clone(); // ECMLI
    cm.flags |= crate::cm::CM_EXPTAIL_STATS;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden from C (original/src via rng_dump helper, seed 181):
    //   SEQ1: 323113200122313232121312331121203202212131202122103310103023
    //   SEQ2: 201100221003221212010132020002202032301001021301002323303201
    #[test]
    fn rng_sample_matches_c() {
        let ghmm = create_genomic_hmm(4);
        assert_eq!(ghmm.nstates, 5);
        let mut r = EslRandom::new(181);
        r.init(181); // generate_sequences reseed
        let s1 = sample_genomic_sequence_from_hmm(&mut r, &ghmm, 60);
        let s2 = sample_genomic_sequence_from_hmm(&mut r, &ghmm, 60);
        let d1: String = (1..=60).map(|x| (s1[x] + b'0') as char).collect();
        let d2: String = (1..=60).map(|x| (s2[x] + b'0') as char).collect();
        assert_eq!(d1, "323113200122313232121312331121203202212131202122103310103023");
        assert_eq!(d2, "201100221003221212010132020002202032301001021301002323303201");
    }

    // End-to-end golden vs C `cmcalibrate --cpu 0 --seed 181 -L 0.02` on
    // TRNAinf-euk-SeC.cm. Byte-identical to C's ECMxx lines:
    //   ECMGC 0.47187 -47.91119 -37.75828  602  ;  ECMGI 0.42183 -45.44548 -34.24703  563
    //   ECMLC 0.97530  -4.23060   2.30109 8764  ;  ECMLI 0.81103  -5.23147   2.53697 8172
    // #[ignore]d (heavy: 2×10 Kb scans ×4 modes) and skips if the fixture is
    // absent. Run with: cargo test --lib -- --ignored calibrate_golden
    #[test]
    #[ignore]
    fn calibrate_golden_matches_c() {
        let path = "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk-SeC.cm";
        if !std::path::Path::new(path).exists() {
            eprintln!("fixture missing, skipping: {path}");
            return;
        }
        let cm = crate::cm_file::cm_file_read_global(path).expect("read CM");
        let cfg = CalibrateConfig {
            seed: 181,
            l_mb: 0.02,
            ..Default::default()
        };
        let res = calibrate_cm(&cm, &cfg).expect("calibrate");
        // (mode, lambda, mu_orig, nrandhits), 5-decimal rounding as in the file.
        let expect = [
            (EXP_CM_GC, 0.47187, -37.75828, 602),
            (EXP_CM_GI, 0.42183, -34.24703, 563),
            (EXP_CM_LC, 0.97530, 2.30109, 8764),
            (EXP_CM_LI, 0.81103, 2.53697, 8172),
        ];
        for (m, lam, mu_orig, nh) in expect {
            let e = &res.exp[m];
            assert_eq!(e.nrandhits, nh, "nrandhits mode {m}");
            assert!((e.lambda - lam).abs() < 5e-6, "lambda mode {m}: {} vs {}", e.lambda, lam);
            assert!(
                (e.mu_orig - mu_orig).abs() < 5e-6,
                "mu_orig mode {m}: {} vs {}",
                e.mu_orig,
                mu_orig
            );
        }
    }
}
