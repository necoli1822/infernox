#![allow(non_snake_case)]
//! prior.rs — faithful byte-for-byte port of Infernal's Dirichlet priors.
//!
//! Ported 1:1 from:
//!   - infernal/src/prior.c      (Prior_Create, PriorifyCM, Prior_Default)
//!   - infernal/src/prior.h      (Prior_t struct)
//!   - easel/esl_mixdchlet.c     (esl_mixdchlet_Create, esl_mixdchlet_MPParameters, mixdchlet_postq)
//!   - easel/esl_dirichlet.c     (esl_dirichlet_logpdf_c)
//!   - easel/esl_stats.c         (esl_stats_LogGamma)
//!   - easel/esl_vectorops.c     (esl_vec_D{Set,Sum,Norm,Exp,Increment,Max,LogSum,LogNorm})
//!
//! All prior math is done in f64 (C `double`), converting to/from f32 only at the
//! cm.t/cm.e boundary, exactly as C does. Kahan summation is preserved in DSum.
//!
//! NOTE on one interface deviation: `Mixdchlet::mp_parameters` takes `&self`
//! (not `&mut self`). In C, `esl_mixdchlet_MPParameters` uses `dchl->postq` as an
//! internal workspace (write-then-read within the same call; never read afterward),
//! which is why C says "you can't declare dchl const". Here that workspace is a
//! local buffer instead, so `mp_parameters` is `&self`. This is numerically
//! identical and lets `priorify_cm(&Prior)` call it through a shared reference,
//! matching C's `PriorifyCM(const Prior_t *pri)`.

use crate::cm::{CM, CM_RSEARCHEMIT};
use crate::constants::{
    B_ST, E_ST, IL_ST, IR_ST, MP_ST,
    MATL_ML, MATP_ML, MATP_MR, MATR_MR,
    // unique state ids (tsetmap first index)
    ROOT_S, ROOT_IL, ROOT_IR, BEGL_S, BEGR_S, BEGR_IL,
    MATP_MP, MATP_D, MATP_IL, MATP_IR,
    MATL_D, MATL_IL, MATR_D, MATR_IR,
    // node types (tsetmap second index); C uses lowercase _nd aliases with same values
    BIF_ND, END_ND, MATL_ND, MATP_ND, MATR_ND,
};

// =============================================================================
// Private f64 vector helpers — faithful ports from easel/esl_vectorops.c.
// Each takes (slice, n) and operates over [0..n), exactly as the C (vec, n) API.
// =============================================================================

/* esl_vec_DSet(): for (i=0;i<n;i++) vec[i] = value; */
fn esl_vec_dset(vec: &mut [f64], n: usize, value: f64) {
    for i in 0..n {
        vec[i] = value;
    }
}

/* esl_vec_DIncrement(): for (i=0;i<n;i++) v[i] += x; */
fn esl_vec_dincrement(v: &mut [f64], n: usize, x: f64) {
    for i in 0..n {
        v[i] += x;
    }
}

/* esl_vec_DSum(): Kahan-compensated summation.
 *   c = 0.0;
 *   for (i=0;i<n;i++) { y = vec[i]-c; t = sum+y; c = (t-sum)-y; sum = t; }
 */
fn esl_vec_dsum(vec: &[f64], n: usize) -> f64 {
    let mut sum: f64 = 0.0;
    let mut c: f64 = 0.0;
    for i in 0..n {
        let y = vec[i] - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/* esl_vec_DMax(): best = vec[0]; for (i=1;i<n;i++) if (vec[i]>best) best=vec[i]; */
fn esl_vec_dmax(vec: &[f64], n: usize) -> f64 {
    let mut best = vec[0];
    for i in 1..n {
        if vec[i] > best {
            best = vec[i];
        }
    }
    best
}

/* esl_vec_DExp(): for (i=0;i<n;i++) vec[i] = exp(vec[i]); */
fn esl_vec_dexp(vec: &mut [f64], n: usize) {
    for i in 0..n {
        vec[i] = vec[i].exp();
    }
}

/* esl_vec_DNorm():
 *   sum = esl_vec_DSum(vec,n);
 *   if (sum != 0.0) for (i..) vec[i] /= sum;
 *   else            for (i..) vec[i] = 1./(double)n;
 */
fn esl_vec_dnorm(vec: &mut [f64], n: usize) {
    let sum = esl_vec_dsum(vec, n);
    if sum != 0.0 {
        for i in 0..n {
            vec[i] /= sum;
        }
    } else {
        for i in 0..n {
            vec[i] = 1.0 / (n as f64);
        }
    }
}

/* esl_vec_DLogSum():
 *   max = esl_vec_DMax(vec,n);
 *   if (max == eslINFINITY) return eslINFINITY;
 *   sum = 0.0;
 *   for (i..) if (vec[i] > max - 500.) sum += exp(vec[i]-max);
 *   sum = log(sum) + max;
 */
fn esl_vec_dlogsum(vec: &[f64], n: usize) -> f64 {
    let max = esl_vec_dmax(vec, n);
    if max == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut sum: f64 = 0.0;
    for i in 0..n {
        if vec[i] > max - 500.0 {
            sum += (vec[i] - max).exp();
        }
    }
    sum.ln() + max
}

/* esl_vec_DLogNorm():
 *   denom = esl_vec_DLogSum(vec,n);
 *   esl_vec_DIncrement(vec,n,-1.*denom);
 *   esl_vec_DExp (vec,n);
 *   esl_vec_DNorm(vec,n);
 */
fn esl_vec_dlognorm(vec: &mut [f64], n: usize) {
    let denom = esl_vec_dlogsum(vec, n);
    esl_vec_dincrement(vec, n, -1.0 * denom);
    esl_vec_dexp(vec, n);
    esl_vec_dnorm(vec, n);
}

// =============================================================================
// esl_stats_LogGamma() — faithful port from easel/esl_stats.c.
// Lanczos-style approximation w/ the exact cof[11] table. Byte parity depends
// on this. Assumes x > 0 (C throws eslERANGE for x <= 0).
// =============================================================================
fn esl_stats_loggamma(x: f64) -> f64 {
    /* static double cof[11] = { ... }; */
    const COF: [f64; 11] = [
        4.694580336184385e+04,
        -1.560605207784446e+05,
        2.065049568014106e+05,
        -1.388934775095388e+05,
        5.031796415085709e+04,
        -9.601592329182778e+03,
        8.785855930895250e+02,
        -3.155153906098611e+01,
        2.908143421162229e-01,
        -2.319827630494973e-04,
        1.251639670050933e-10,
    ];

    /* xx = x - 1.0;  tx = tmp = xx + 11.0;  value = 1.0; */
    let xx = x - 1.0;
    let mut tmp = xx + 11.0;
    let mut tx = xx + 11.0;
    let mut value = 1.0_f64;
    /* for (i=10;i>=0;i--) { value += cof[i]/tmp; tmp -= 1.0; } */
    for i in (0..=10).rev() {
        value += COF[i] / tmp;
        tmp -= 1.0;
    }
    /* value = log(value); tx += 0.5;
     * value += 0.918938533 + (xx+0.5)*log(tx) - tx; */
    value = value.ln();
    tx += 0.5;
    value += 0.918938533 + (xx + 0.5) * tx.ln() - tx;
    value
}

// =============================================================================
// esl_dirichlet_logpdf_c() — faithful port from easel/esl_dirichlet.c.
// log P(c | alpha) for a single Dirichlet.
// =============================================================================
fn esl_dirichlet_logpdf_c(c: &[f64], alpha: &[f64], k: usize) -> f64 {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let mut logp = 0.0;
    for a in 0..k {
        sum1 += c[a] + alpha[a];
        sum2 += alpha[a];
        sum3 += c[a];
        let a1 = esl_stats_loggamma(alpha[a] + c[a]);
        let a2 = esl_stats_loggamma(c[a] + 1.0);
        let a3 = esl_stats_loggamma(alpha[a]);
        logp += a1 - a2 - a3;
    }
    let a1 = esl_stats_loggamma(sum1);
    let a2 = esl_stats_loggamma(sum2);
    let a3 = esl_stats_loggamma(sum3 + 1.0);
    logp += a2 + a3 - a1;
    logp
}

// =============================================================================
// ESL_MIXDCHLET object — faithful port from easel/esl_mixdchlet.c.
// =============================================================================

/// Mixture Dirichlet: `Q` components, each with `K` parameters.
#[derive(Clone)]
pub struct Mixdchlet {
    pub Q: usize,
    pub K: usize,
    pub q: Vec<f64>,
    pub alpha: Vec<Vec<f64>>,
    pub postq: Vec<f64>,
}

impl Mixdchlet {
    /* esl_mixdchlet_Create(Q, K): allocates q[Q], postq[Q], alpha[Q][K].
     * Easel zeroes the allocations (esl_mat_DCreate); the caller then fills
     * q[]/alpha[][]. We mirror that with zeroed Vecs. */
    pub fn create(Q: usize, K: usize) -> Self {
        Mixdchlet {
            Q,
            K,
            q: vec![0.0; Q],
            alpha: vec![vec![0.0; K]; Q],
            postq: vec![0.0; Q],
        }
    }

    /* mixdchlet_postq(): P(q | c), posterior prob of each component.
     *   for (k=0;k<Q;k++)
     *     if (q[k] > 0.) postq[k] = log(q[k]) + esl_dirichlet_logpdf_c(c, alpha[k], K);
     *     else           postq[k] = -eslINFINITY;
     *   esl_vec_DLogNorm(postq, Q);
     * Here `postq` is a caller-provided workspace (see module NOTE). */
    fn mixdchlet_postq(&self, c: &[f64], postq: &mut [f64]) {
        for k in 0..self.Q {
            if self.q[k] > 0.0 {
                postq[k] = self.q[k].ln() + esl_dirichlet_logpdf_c(c, &self.alpha[k], self.K);
            } else {
                postq[k] = -f64::INFINITY;
            }
        }
        esl_vec_dlognorm(postq, self.Q);
    }

    /* esl_mixdchlet_MPParameters(): mean posterior parameter estimates.
     *   mixdchlet_postq(dchl, c);
     *   totc = esl_vec_DSum(c, K);
     *   esl_vec_DSet(p, K, 0.);
     *   for (k=0;k<Q;k++) {
     *     totalpha = esl_vec_DSum(alpha[k], K);
     *     for (a=0;a<K;a++) p[a] += postq[k] * (c[a]+alpha[k][a]) / (totc+totalpha);
     *   }
     *   esl_vec_DNorm(p, K);
     */
    pub fn mp_parameters(&self, c: &[f64], p: &mut [f64]) {
        let mut postq = vec![0.0f64; self.Q];
        self.mixdchlet_postq(c, &mut postq);

        let totc = esl_vec_dsum(c, self.K);
        esl_vec_dset(p, self.K, 0.0);
        for k in 0..self.Q {
            let totalpha = esl_vec_dsum(&self.alpha[k], self.K);
            for a in 0..self.K {
                p[a] += postq[k] * (c[a] + self.alpha[k][a]) / (totc + totalpha);
            }
        }
        /* should be normalized already, but for good measure: */
        esl_vec_dnorm(p, self.K);
    }
}

// =============================================================================
// Prior_t struct — faithful port from prior.h.
// =============================================================================

/// Dirichlet priors on all model parameters.
pub struct Prior {
    /// number of transition sets
    pub tsetnum: usize,
    /// tsetmap[a][b]: transition set index from unique state `a` to node type `b`
    pub tsetmap: [[i32; 8]; 21], // [UNIQUESTATES][NODETYPES]
    /// array of transition priors, 0..tsetnum-1
    pub t: Vec<Mixdchlet>,
    /// consensus base pair emission prior
    pub mbp: Mixdchlet,
    /// consensus singlet emission prior
    pub mnt: Mixdchlet,
    /// nonconsensus singlet emission prior
    pub i: Mixdchlet,
    /// maximum # of components in any prior
    pub maxnq: usize,
    /// maximum # of parameters in any prior
    pub maxnalpha: usize,
}

/* helper: build a single-component (Q=1) Dirichlet from its alpha vector, with
 * q[0] = 1.0. Mirrors the autogenerated pattern:
 *   pri->t[i] = esl_mixdchlet_Create(1, K);
 *   pri->t[i]->q[0] = 1.0;
 *   pri->t[i]->alpha[0][a] = ...;   (a = 0..K-1)
 */
fn t1(alphas: &[f64]) -> Mixdchlet {
    let mut d = Mixdchlet::create(1, alphas.len());
    d.q[0] = 1.0;
    for (a, &val) in alphas.iter().enumerate() {
        d.alpha[0][a] = val;
    }
    d
}

// =============================================================================
// Prior_Default() — faithful port from prior.c (Prior_Default, l.330).
//
// Builds the default post-1.0.2 mixture Dirichlet prior (the 'p33' prior,
// trained on 82 Rfam 10.0 seeds). The 74 transition sets and the 10-component
// base-pair prior `mbp` are common to both mimic_h3 cases. When `mimic_h3` is
// FALSE, `mnt`/`i` are 10-component nucleotide priors. When TRUE (used for
// models with 0 basepairs), transitions out of MATL nodes (t[32],t[37],t[42])
// are overwritten and `mnt`/`i` use the H3 nucleotide priors.
// =============================================================================
pub fn prior_default(mimic_h3: bool) -> Prior {
    // Prior_Create(): tsetmap all -1, counters 0.
    let mut tsetmap = [[-1i32; 8]; 21];

    // pri->tsetnum = 74;
    let mut t: Vec<Mixdchlet> = Vec::with_capacity(74);

    /*****************************************************************
     * Autogenerated transition-set block (prifile2code.pl mixture.pri).
     * Each entry: pri->tsetmap[<uniqstate>][<node>] = <i>; pri->t[<i>] = ...
     * Transcribed value-for-value; t[] pushed in index order 0..73.
     *****************************************************************/

    // t[0]  MATP_MP -> BIF
    tsetmap[MATP_MP as usize][BIF_ND as usize] = 0;
    t.push(t1(&[0.067710091654, 0.000047753225, 0.483183211040]));
    // t[1]  MATP_MP -> END
    tsetmap[MATP_MP as usize][END_ND as usize] = 1;
    t.push(t1(&[0.067710091654, 0.000047753225, 0.483183211040]));
    // t[2]  MATP_MP -> MATL
    tsetmap[MATP_MP as usize][MATL_ND as usize] = 2;
    t.push(t1(&[0.028518011579, 0.024705844026, 1.464047470747, 0.074164509948]));
    // t[3]  MATP_MP -> MATP
    tsetmap[MATP_MP as usize][MATP_ND as usize] = 3;
    t.push(t1(&[
        0.016729608598, 0.017449035307, 7.164604225972, 0.040744980202, 0.033562178957,
        0.025523202345,
    ]));
    // t[4]  MATP_MP -> MATR
    tsetmap[MATP_MP as usize][MATR_ND as usize] = 4;
    t.push(t1(&[0.032901537296, 0.013876834787, 1.694917068307, 0.162141225286]));

    // t[5]  MATP_ML -> BIF
    tsetmap[MATP_ML as usize][BIF_ND as usize] = 5;
    t.push(t1(&[1.0, 1.0, 1.0]));
    // t[6]  MATP_ML -> END
    tsetmap[MATP_ML as usize][END_ND as usize] = 6;
    t.push(t1(&[1.0, 1.0, 1.0]));
    // t[7]  MATP_ML -> MATL
    tsetmap[MATP_ML as usize][MATL_ND as usize] = 7;
    t.push(t1(&[0.068859974656, 0.060683472648, 0.655691547663, 0.146392271070]));
    // t[8]  MATP_ML -> MATP
    tsetmap[MATP_ML as usize][MATP_ND as usize] = 8;
    t.push(t1(&[
        0.009119452604, 0.007174198989, 0.279841652851, 0.345855381430, 0.007961193216,
        0.044123881735,
    ]));
    // t[9]  MATP_ML -> MATR
    tsetmap[MATP_ML as usize][MATR_ND as usize] = 9;
    t.push(t1(&[0.061640259819, 0.014142411829, 0.133564345209, 0.117860328247]));

    // t[10] MATP_MR -> BIF
    tsetmap[MATP_MR as usize][BIF_ND as usize] = 10;
    t.push(t1(&[1.0, 1.0, 1.0]));
    // t[11] MATP_MR -> END
    tsetmap[MATP_MR as usize][END_ND as usize] = 11;
    t.push(t1(&[1.0, 1.0, 1.0]));
    // t[12] MATP_MR -> MATL
    tsetmap[MATP_MR as usize][MATL_ND as usize] = 12;
    t.push(t1(&[0.024723293475, 0.048463880304, 0.212532685951, 0.407547325080]));
    // t[13] MATP_MR -> MATP
    tsetmap[MATP_MR as usize][MATP_ND as usize] = 13;
    t.push(t1(&[
        0.006294030132, 0.015189408169, 0.258896467198, 0.015420910305, 0.449746529026,
        0.053194553636,
    ]));
    // t[14] MATP_MR -> MATR
    tsetmap[MATP_MR as usize][MATR_ND as usize] = 14;
    t.push(t1(&[0.020819322736, 0.000060497356, 0.272689176849, 0.063856784928]));

    // t[15] MATP_D -> BIF
    tsetmap[MATP_D as usize][BIF_ND as usize] = 15;
    t.push(t1(&[1.0, 1.0, 1.0]));
    // t[16] MATP_D -> END
    tsetmap[MATP_D as usize][END_ND as usize] = 16;
    t.push(t1(&[1.0, 1.0, 1.0]));
    // t[17] MATP_D -> MATL
    tsetmap[MATP_D as usize][MATL_ND as usize] = 17;
    t.push(t1(&[0.024577940691, 0.030655567559, 0.121290355765, 0.406621701238]));
    // t[18] MATP_D -> MATP
    tsetmap[MATP_D as usize][MATP_ND as usize] = 18;
    t.push(t1(&[
        0.001029025955, 0.002536729756, 0.046719556839, 0.029117903291, 0.028767509361,
        0.436842892057,
    ]));
    // t[19] MATP_D -> MATR
    tsetmap[MATP_D as usize][MATR_ND as usize] = 19;
    t.push(t1(&[0.000017041108, 0.000007069171, 0.028384306256, 0.087965488640]));

    // t[20] MATP_IL -> BIF
    tsetmap[MATP_IL as usize][BIF_ND as usize] = 20;
    t.push(t1(&[0.943443048986, 0.064001237265, 0.432230812455]));
    // t[21] MATP_IL -> END
    tsetmap[MATP_IL as usize][END_ND as usize] = 21;
    t.push(t1(&[0.943443048986, 0.064001237265, 0.432230812455]));
    // t[22] MATP_IL -> MATL
    tsetmap[MATP_IL as usize][MATL_ND as usize] = 22;
    t.push(t1(&[0.250101882938, 0.155728904821, 0.370945030932, 0.027811408475]));
    // t[23] MATP_IL -> MATP
    tsetmap[MATP_IL as usize][MATP_ND as usize] = 23;
    t.push(t1(&[
        0.157307265492, 0.131105492208, 0.555106727689, 0.041624804903, 0.024305424386,
        0.030756705205,
    ]));
    // t[24] MATP_IL -> MATR
    tsetmap[MATP_IL as usize][MATR_ND as usize] = 24;
    t.push(t1(&[0.155093374292, 0.054734614999, 0.714409186001, 0.168407110635]));

    // t[25] MATP_IR -> BIF
    tsetmap[MATP_IR as usize][BIF_ND as usize] = 25;
    t.push(t1(&[0.264643213319, 0.671462565227]));
    // t[26] MATP_IR -> END
    tsetmap[MATP_IR as usize][END_ND as usize] = 26;
    t.push(t1(&[0.264643213319, 0.671462565227]));
    // t[27] MATP_IR -> MATL
    tsetmap[MATP_IR as usize][MATL_ND as usize] = 27;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691]));
    // t[28] MATP_IR -> MATP
    tsetmap[MATP_IR as usize][MATP_ND as usize] = 28;
    t.push(t1(&[
        0.291829430523, 1.098441427679, 0.025595408318, 0.091146313822, 0.042349119486,
    ]));
    // t[29] MATP_IR -> MATR
    tsetmap[MATP_IR as usize][MATR_ND as usize] = 29;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204]));

    // t[30] MATL_ML -> BIF
    tsetmap[MATL_ML as usize][BIF_ND as usize] = 30;
    t.push(t1(&[0.009635966745, 1.220143960207]));
    // t[31] MATL_ML -> END
    tsetmap[MATL_ML as usize][END_ND as usize] = 31;
    t.push(t1(&[0.009635966745, 1.220143960207]));
    // t[32] MATL_ML -> MATL
    tsetmap[MATL_ML as usize][MATL_ND as usize] = 32;
    t.push(t1(&[0.015185708311, 1.809432933023, 0.038601480352]));
    // t[33] MATL_ML -> MATP
    tsetmap[MATL_ML as usize][MATP_ND as usize] = 33;
    t.push(t1(&[
        0.031820644019, 2.300193431878, 0.036163737927, 0.031218244200, 0.016826710214,
    ]));
    // t[34] MATL_ML -> MATR
    tsetmap[MATL_ML as usize][MATR_ND as usize] = 34;
    t.push(t1(&[0.012395245929, 2.076134487839, 0.039781067793]));

    // t[35] MATL_D -> BIF
    tsetmap[MATL_D as usize][BIF_ND as usize] = 35;
    t.push(t1(&[0.019509171372, 6.781321301695]));
    // t[36] MATL_D -> END
    tsetmap[MATL_D as usize][END_ND as usize] = 36;
    t.push(t1(&[0.019509171372, 6.781321301695]));
    // t[37] MATL_D -> MATL
    tsetmap[MATL_D as usize][MATL_ND as usize] = 37;
    t.push(t1(&[0.005679808868, 0.127365862719, 0.277086556814]));
    // t[38] MATL_D -> MATP
    tsetmap[MATL_D as usize][MATP_ND as usize] = 38;
    t.push(t1(&[
        0.023424968753, 0.417640407951, 0.039088991906, 0.120577442402, 0.128103786646,
    ]));
    // t[39] MATL_D -> MATR
    tsetmap[MATL_D as usize][MATR_ND as usize] = 39;
    t.push(t1(&[0.013699691994, 0.405128575339, 0.254775565405]));

    // t[40] MATL_IL -> BIF
    tsetmap[MATL_IL as usize][BIF_ND as usize] = 40;
    t.push(t1(&[0.264643213319, 0.671462565227]));
    // t[41] MATL_IL -> END
    tsetmap[MATL_IL as usize][END_ND as usize] = 41;
    t.push(t1(&[0.264643213319, 0.671462565227]));
    // t[42] MATL_IL -> MATL
    tsetmap[MATL_IL as usize][MATL_ND as usize] = 42;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691]));
    // t[43] MATL_IL -> MATP
    tsetmap[MATL_IL as usize][MATP_ND as usize] = 43;
    t.push(t1(&[
        0.291829430523, 1.098441427679, 0.091146313822, 0.025595408318, 0.042349119486,
    ]));
    // t[44] MATL_IL -> MATR
    tsetmap[MATL_IL as usize][MATR_ND as usize] = 44;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204]));

    // t[45] MATR_MR -> BIF
    tsetmap[MATR_MR as usize][BIF_ND as usize] = 45;
    t.push(t1(&[0.009635966745, 1.220143960207]));
    // t[46] MATR_MR -> MATP
    tsetmap[MATR_MR as usize][MATP_ND as usize] = 46;
    t.push(t1(&[
        0.031820644019, 2.300193431878, 0.036163737927, 0.031218244200, 0.016826710214,
    ]));
    // t[47] MATR_MR -> MATR
    tsetmap[MATR_MR as usize][MATR_ND as usize] = 47;
    t.push(t1(&[0.012395245929, 2.076134487839, 0.039781067793]));

    // t[48] MATR_D -> BIF
    tsetmap[MATR_D as usize][BIF_ND as usize] = 48;
    t.push(t1(&[0.021604946951, 0.444765555211]));
    // t[49] MATR_D -> MATP
    tsetmap[MATR_D as usize][MATP_ND as usize] = 49;
    t.push(t1(&[
        0.021273745319, 0.532292228853, 0.110249350652, 0.040890357850, 0.164194410420,
    ]));
    // t[50] MATR_D -> MATR
    tsetmap[MATR_D as usize][MATR_ND as usize] = 50;
    t.push(t1(&[0.005806440507, 0.164264844267, 0.316876127883]));

    // t[51] MATR_IR -> BIF
    tsetmap[MATR_IR as usize][BIF_ND as usize] = 51;
    t.push(t1(&[0.264643213319, 0.671462565227]));
    // t[52] MATR_IR -> MATP
    tsetmap[MATR_IR as usize][MATP_ND as usize] = 52;
    t.push(t1(&[
        0.291829430523, 1.098441427679, 0.025595408318, 0.091146313822, 0.042349119486,
    ]));
    // t[53] MATR_IR -> MATR
    tsetmap[MATR_IR as usize][MATR_ND as usize] = 53;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204]));

    // t[54] BEGL_S -> BIF
    tsetmap[BEGL_S as usize][BIF_ND as usize] = 54;
    t.push(t1(&[1.0]));
    // t[55] BEGL_S -> MATP
    tsetmap[BEGL_S as usize][MATP_ND as usize] = 55;
    t.push(t1(&[4.829712747509, 0.061131109227, 0.092185242101, 0.059154827887]));

    // t[56] BEGR_S -> BIF
    tsetmap[BEGR_S as usize][BIF_ND as usize] = 56;
    t.push(t1(&[0.009635966745, 1.220143960207]));
    // t[57] BEGR_S -> MATL
    tsetmap[BEGR_S as usize][MATL_ND as usize] = 57;
    t.push(t1(&[0.015185708311, 1.809432933023, 0.038601480352]));
    // t[58] BEGR_S -> MATP
    tsetmap[BEGR_S as usize][MATP_ND as usize] = 58;
    t.push(t1(&[
        0.031820644019, 2.300193431878, 0.036163737927, 0.031218244200, 0.016826710214,
    ]));

    // t[59] BEGR_IL -> BIF
    tsetmap[BEGR_IL as usize][BIF_ND as usize] = 59;
    t.push(t1(&[0.264643213319, 0.671462565227]));
    // t[60] BEGR_IL -> MATL
    tsetmap[BEGR_IL as usize][MATL_ND as usize] = 60;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691]));
    // t[61] BEGR_IL -> MATP
    tsetmap[BEGR_IL as usize][MATP_ND as usize] = 61;
    t.push(t1(&[
        0.291829430523, 1.098441427679, 0.091146313822, 0.025595408318, 0.042349119486,
    ]));

    // t[62] ROOT_S -> BIF
    tsetmap[ROOT_S as usize][BIF_ND as usize] = 62;
    t.push(t1(&[0.067710091654, 0.000047753225, 0.483183211040]));
    // t[63] ROOT_S -> MATL
    tsetmap[ROOT_S as usize][MATL_ND as usize] = 63;
    t.push(t1(&[0.028518011579, 0.024705844026, 1.464047470747, 0.074164509948]));
    // t[64] ROOT_S -> MATP
    tsetmap[ROOT_S as usize][MATP_ND as usize] = 64;
    t.push(t1(&[
        0.016729608598, 0.017449035307, 7.164604225972, 0.040744980202, 0.033562178957,
        0.025523202345,
    ]));
    // t[65] ROOT_S -> MATR
    tsetmap[ROOT_S as usize][MATR_ND as usize] = 65;
    t.push(t1(&[0.032901537296, 0.013876834787, 1.694917068307, 0.162141225286]));

    // t[66] ROOT_IL -> BIF
    tsetmap[ROOT_IL as usize][BIF_ND as usize] = 66;
    t.push(t1(&[0.943443048986, 0.064001237265, 0.432230812455]));
    // t[67] ROOT_IL -> MATL
    tsetmap[ROOT_IL as usize][MATL_ND as usize] = 67;
    t.push(t1(&[0.250101882938, 0.155728904821, 0.370945030932, 0.027811408475]));
    // t[68] ROOT_IL -> MATP
    tsetmap[ROOT_IL as usize][MATP_ND as usize] = 68;
    t.push(t1(&[
        0.157307265492, 0.131105492208, 0.555106727689, 0.041624804903, 0.024305424386,
        0.030756705205,
    ]));
    // t[69] ROOT_IL -> MATR
    tsetmap[ROOT_IL as usize][MATR_ND as usize] = 69;
    t.push(t1(&[0.155093374292, 0.054734614999, 0.714409186001, 0.168407110635]));

    // t[70] ROOT_IR -> BIF
    tsetmap[ROOT_IR as usize][BIF_ND as usize] = 70;
    t.push(t1(&[0.264643213319, 0.671462565227]));
    // t[71] ROOT_IR -> MATL
    tsetmap[ROOT_IR as usize][MATL_ND as usize] = 71;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691]));
    // t[72] ROOT_IR -> MATP
    tsetmap[ROOT_IR as usize][MATP_ND as usize] = 72;
    t.push(t1(&[
        0.291829430523, 1.098441427679, 0.025595408318, 0.091146313822, 0.042349119486,
    ]));
    // t[73] ROOT_IR -> MATR
    tsetmap[ROOT_IR as usize][MATR_ND as usize] = 73;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204]));

    /*****************************************************************
     * Consensus base pair emission prior mbp = esl_mixdchlet_Create(10, 16).
     * Common to both mimic_h3 cases. (prior.c l.904-1084)
     *****************************************************************/
    let mut mbp = Mixdchlet::create(10, 16);
    // q[0]
    mbp.q[0] = 0.016584;
    mbp.alpha[0] = vec![
        0.142252, 0.180113, 0.153776, 0.222524, 0.539721, 0.170380, 0.230123, 0.190004, 0.583682,
        0.222992, 0.134277, 0.187596, 12172.002267, 0.546735, 14.841962, 17.271555,
    ];
    // q[1]
    mbp.q[1] = 0.000948;
    mbp.alpha[1] = vec![
        2.547410, 14.293143, 0.015263, 7.761130, 0.029915, 3.493007, 14.049507, 1.480341, 3.754643,
        7.140983, 4.733217, 44.190624, 2.758417, 1.687945, 2.421882, 2.724272,
    ];
    // q[2]
    mbp.q[2] = 0.185395;
    mbp.alpha[2] = vec![
        0.054512, 0.067070, 0.054506, 1.210822, 0.119647, 0.030366, 3.188992, 0.076098, 0.060153,
        1.426299, 0.042134, 0.362385, 2.308941, 0.048026, 0.695604, 0.146166,
    ];
    // q[3]
    mbp.q[3] = 0.082929;
    mbp.alpha[3] = vec![
        0.481661, 0.414811, 0.419836, 3.024237, 0.421853, 0.232594, 3.637964, 0.328914, 0.400575,
        2.647559, 0.269173, 1.022533, 3.376215, 0.380735, 1.397263, 0.695235,
    ];
    // q[4]
    mbp.q[4] = 0.039651;
    mbp.alpha[4] = vec![
        0.145102, 0.122876, 3.107999, 0.099093, 10.564395, 4.450523, 15500.159054, 0.049506,
        0.032312, 0.368096, 4.375012, 0.111267, 0.904329, 0.074819, 0.376995, 0.093335,
    ];
    // q[5]
    mbp.q[5] = 0.141227;
    mbp.alpha[5] = vec![
        0.016163, 0.040913, 0.014116, 0.527169, 0.003200, 0.001437, 0.074551, 0.013706, 0.019149,
        0.413953, 0.012037, 0.268400, 0.078554, 0.005712, 0.020960, 0.037299,
    ];
    // q[6]
    mbp.q[6] = 0.132571;
    mbp.alpha[6] = vec![
        0.004230, 0.045568, 0.000699, 0.258190, 0.001391, 0.020574, 0.073793, 0.001111, 0.017598,
        7.014687, 0.015465, 0.189706, 0.057044, 0.014820, 0.013400, 0.001169,
    ];
    // q[7]
    mbp.q[7] = 0.249417;
    mbp.alpha[7] = vec![
        0.008027, 0.006602, 0.012884, 0.089652, 0.040423, 0.011659, 0.789524, 0.021433, 0.011990,
        0.091424, 0.011034, 0.019953, 0.389278, 0.009006, 0.198688, 0.027512,
    ];
    // q[8]
    mbp.q[8] = 0.140727;
    mbp.alpha[8] = vec![
        0.068663, 0.176455, 0.077881, 2.165192, 0.035566, 0.051544, 1.087382, 0.048265, 0.057469,
        5.631915, 0.048459, 0.906756, 0.904423, 0.086167, 0.270333, 0.159528,
    ];
    // q[9]
    mbp.q[9] = 0.010551;
    mbp.alpha[9] = vec![
        0.478576, 0.402540, 18.466281, 16947.982248, 0.389092, 0.386664, 0.619656, 20.908826,
        0.375696, 4.605442, 0.396373, 13.623423, 0.513956, 0.363145, 0.606193, 18.301915,
    ];

    // mnt and i depend on mimic_h3.
    let mnt;
    let i;
    if !mimic_h3 {
        /* normal case: 10-component nucleotide priors (prior.c l.1086-1206) */
        let mut m = Mixdchlet::create(10, 4);
        m.q[0] = 0.081706;
        m.alpha[0] = vec![0.963855, 3.273863, 0.444739, 1.958731];
        m.q[1] = 0.104534;
        m.alpha[1] = vec![0.589011, 0.648423, 0.360672, 5.771004];
        m.q[2] = 0.048944;
        m.alpha[2] = vec![2.609834, 0.127100, 1.180559, 0.134264];
        m.q[3] = 0.064111;
        m.alpha[3] = vec![1.259286, 0.659029, 4.874613, 0.882126];
        m.q[4] = 0.085266;
        m.alpha[4] = vec![4.664219, 0.628128, 0.448894, 0.661556];
        m.q[5] = 0.045348;
        m.alpha[5] = vec![0.250974, 9.700414, 0.206184, 0.338607];
        m.q[6] = 0.100949;
        m.alpha[6] = vec![0.178455, 0.049385, 7.914643, 0.100802];
        m.q[7] = 0.108835;
        m.alpha[7] = vec![23.818220, 0.064454, 0.119891, 0.101866];
        m.q[8] = 0.234814;
        m.alpha[8] = vec![2.980233, 1.817786, 1.818483, 3.042635];
        m.q[9] = 0.125493;
        m.alpha[9] = vec![0.024428, 0.064315, 0.008054, 0.107062];
        mnt = m;

        let mut ii = Mixdchlet::create(10, 4);
        ii.q[0] = 0.081706;
        ii.alpha[0] = vec![0.963855, 3.273863, 0.444739, 1.958731];
        ii.q[1] = 0.104534;
        ii.alpha[1] = vec![0.589011, 0.648423, 0.360672, 5.771004];
        ii.q[2] = 0.048944;
        ii.alpha[2] = vec![2.609834, 0.127100, 1.180559, 0.134264];
        ii.q[3] = 0.064111;
        ii.alpha[3] = vec![1.259286, 0.659029, 4.874613, 0.882126];
        ii.q[4] = 0.085266;
        ii.alpha[4] = vec![4.664219, 0.628128, 0.448894, 0.661556];
        ii.q[5] = 0.045348;
        ii.alpha[5] = vec![0.250974, 9.700414, 0.206184, 0.338607];
        ii.q[6] = 0.100949;
        ii.alpha[6] = vec![0.178455, 0.049385, 7.914643, 0.100802];
        ii.q[7] = 0.108835;
        ii.alpha[7] = vec![23.818220, 0.064454, 0.119891, 0.101866];
        ii.q[8] = 0.234814;
        ii.alpha[8] = vec![2.980233, 1.817786, 1.818483, 3.042635];
        ii.q[9] = 0.125493;
        ii.alpha[9] = vec![0.024428, 0.064315, 0.008054, 0.107062];
        i = ii;
    } else {
        /* mimic_h3 == TRUE: for models with 0 basepairs. (prior.c l.1210-1272)
         * Copied from hmmer p7_prior_CreateNucleic(). Overwrite MATL-node
         * transitions t[32], t[37], t[42], then set H3 nucleotide mnt and a
         * uninformative insert prior i. */

        // MATL_ML -> MATL, overwrite t[32]
        tsetmap[MATL_ML as usize][MATL_ND as usize] = 32;
        t[32].q[0] = 1.0;
        t[32].alpha[0][0] = 0.1; /* ML->IL */
        t[32].alpha[0][1] = 2.0; /* ML->ML */
        t[32].alpha[0][2] = 0.1; /* ML->D  */

        // MATL_D -> MATL, overwrite t[37]
        tsetmap[MATL_D as usize][MATL_ND as usize] = 37;
        t[37].q[0] = 1.0;
        t[37].alpha[0][0] = 0.0001; /* D->IL (irrelevant; D->I set IMPOSSIBLE later) */
        t[37].alpha[0][1] = 0.1; /* D->ML */
        t[37].alpha[0][2] = 0.2; /* D->D  */

        // MATL_IL -> MATL, overwrite t[42]
        tsetmap[MATL_IL as usize][MATL_ND as usize] = 42;
        t[42].q[0] = 1.0;
        t[42].alpha[0][0] = 0.02; /* IL->IL */
        t[42].alpha[0][1] = 0.006; /* IL->ML */
        t[42].alpha[0][2] = 0.0001; /* IL->D  (irrelevant) */

        // singlet emissions: mnt = esl_mixdchlet_Create(4, 4)
        let mut m = Mixdchlet::create(4, 4);
        m.q[0] = 0.24;
        m.alpha[0] = vec![0.16, 0.45, 0.12, 0.39];
        m.q[1] = 0.26;
        m.alpha[1] = vec![0.09, 0.03, 0.09, 0.04];
        m.q[2] = 0.08;
        m.alpha[2] = vec![1.29, 0.40, 6.58, 0.51];
        m.q[3] = 0.42;
        m.alpha[3] = vec![1.74, 1.49, 1.57, 1.95];
        mnt = m;

        // insert, uninformative: i = esl_mixdchlet_Create(1, 4)
        let mut ii = Mixdchlet::create(1, 4);
        ii.q[0] = 1.0;
        ii.alpha[0] = vec![1.0, 1.0, 1.0, 1.0];
        i = ii;
    }

    // pri->maxnq = 10; pri->maxnalpha = 16;
    Prior {
        tsetnum: 74,
        tsetmap,
        t,
        mbp,
        mnt,
        i,
        maxnq: 10,
        maxnalpha: 16,
    }
}

// =============================================================================
// PriorifyCM() — faithful port from prior.c (l.190-275).
//
// Given a CM containing counts, add Dirichlet pseudocounts and renormalize.
// The Easel Dirichlet routines are f64, the CM is f32, so we convert per-vector.
// =============================================================================
pub fn priorify_cm(cm: &mut CM, pri: &Prior) {
    // cm->abc->K = 4 (RNA)
    const K: usize = 4;

    /* ESL_ALLOC(counts, sizeof(double)*maxnalpha);
     * ESL_ALLOC(probs,  sizeof(double)*maxnalpha);
     * ESL_ALLOC(mixq,   sizeof(double)*maxnq);  -- allocated but unused in calc */
    let mut counts = vec![0.0f64; pri.maxnalpha];
    let mut probs = vec![0.0f64; pri.maxnalpha];
    let _mixq = vec![0.0f64; pri.maxnq];

    for v in 0..(cm.m as usize) {
        /* Priorify transition vector if not a BIF or E state */
        if (cm.sttype[v] as i32) != B_ST && (cm.sttype[v] as i32) != E_ST {
            /* nxtndtype = cm->ndtype[cm->ndidx[cm->cfirst[v] + cm->cnum[v] - 1]];
             * setnum = pri->tsetmap[(int) cm->stid[v]][nxtndtype]; */
            let last = (cm.cfirst[v] + cm.cnum[v] - 1) as usize;
            let nxtndtype = cm.ndtype[cm.ndidx[last] as usize] as i32;
            let setnum = pri.tsetmap[cm.stid[v] as usize][nxtndtype as usize];
            let cnum = cm.cnum[v] as usize;

            for c in 0..cnum {
                counts[c] = cm.t[v][c] as f64;
            }
            pri.t[setnum as usize].mp_parameters(&counts, &mut probs);
            for c in 0..cnum {
                cm.t[v][c] = probs[c] as f32;
            }
        }

        /* in rsearch emit mode, do not priorify emissions */
        if (cm.flags & CM_RSEARCHEMIT) == 0 {
            if (cm.sttype[v] as i32) == MP_ST {
                /* Consensus base pairs: K*K = 16 params */
                for c in 0..(K * K) {
                    counts[c] = cm.e[v][c] as f64;
                }
                pri.mbp.mp_parameters(&counts, &mut probs);
                for c in 0..(K * K) {
                    cm.e[v][c] = probs[c] as f32;
                }
            } else if (cm.stid[v] as i32) == MATL_ML || (cm.stid[v] as i32) == MATR_MR {
                /* Consensus singlets */
                for c in 0..K {
                    counts[c] = cm.e[v][c] as f64;
                }
                pri.mnt.mp_parameters(&counts, &mut probs);
                for c in 0..K {
                    cm.e[v][c] = probs[c] as f32;
                }
            } else if (cm.sttype[v] as i32) == IL_ST
                || (cm.sttype[v] as i32) == IR_ST
                || (cm.stid[v] as i32) == MATP_ML
                || (cm.stid[v] as i32) == MATP_MR
            {
                /* nonconsensus singlets */
                for c in 0..K {
                    counts[c] = cm.e[v][c] as f64;
                }
                pri.i.mp_parameters(&counts, &mut probs);
                for c in 0..K {
                    cm.e[v][c] = probs[c] as f32;
                }
            }
        }
    } /* end loop over states v */
}

/// C: Prior_Default_v0p56_through_v1p02() (prior.c:1298). Default prior from
/// Infernal v0.56 through v1.0.2 (used by --p56 and --v1p0). Autogenerated
/// value-for-value transcription of the prior.c autogen block.
pub fn prior_v0p56_through_v1p02() -> Prior {
    let mut tsetmap = [[-1i32; 8]; 21];
    let mut t: Vec<Mixdchlet> = Vec::with_capacity(74);
    tsetmap[MATP_MP as usize][BIF_ND as usize] = 0;
    t.push(t1(&[0.067710091654, 0.000047753225, 0.483183211040])); // t[0]
    tsetmap[MATP_MP as usize][END_ND as usize] = 1;
    t.push(t1(&[0.067710091654, 0.000047753225, 0.483183211040])); // t[1]
    tsetmap[MATP_MP as usize][MATL_ND as usize] = 2;
    t.push(t1(&[0.028518011579, 0.024705844026, 1.464047470747, 0.074164509948])); // t[2]
    tsetmap[MATP_MP as usize][MATP_ND as usize] = 3;
    t.push(t1(&[0.016729608598, 0.017449035307, 7.164604225972, 0.040744980202, 0.033562178957, 0.025523202345])); // t[3]
    tsetmap[MATP_MP as usize][MATR_ND as usize] = 4;
    t.push(t1(&[0.032901537296, 0.013876834787, 1.694917068307, 0.162141225286])); // t[4]
    tsetmap[MATP_ML as usize][BIF_ND as usize] = 5;
    t.push(t1(&[1.0, 1.0, 1.0])); // t[5]
    tsetmap[MATP_ML as usize][END_ND as usize] = 6;
    t.push(t1(&[1.0, 1.0, 1.0])); // t[6]
    tsetmap[MATP_ML as usize][MATL_ND as usize] = 7;
    t.push(t1(&[0.068859974656, 0.060683472648, 0.655691547663, 0.146392271070])); // t[7]
    tsetmap[MATP_ML as usize][MATP_ND as usize] = 8;
    t.push(t1(&[0.009119452604, 0.007174198989, 0.279841652851, 0.345855381430, 0.007961193216, 0.044123881735])); // t[8]
    tsetmap[MATP_ML as usize][MATR_ND as usize] = 9;
    t.push(t1(&[0.061640259819, 0.014142411829, 0.133564345209, 0.117860328247])); // t[9]
    tsetmap[MATP_MR as usize][BIF_ND as usize] = 10;
    t.push(t1(&[1.0, 1.0, 1.0])); // t[10]
    tsetmap[MATP_MR as usize][END_ND as usize] = 11;
    t.push(t1(&[1.0, 1.0, 1.0])); // t[11]
    tsetmap[MATP_MR as usize][MATL_ND as usize] = 12;
    t.push(t1(&[0.024723293475, 0.048463880304, 0.212532685951, 0.407547325080])); // t[12]
    tsetmap[MATP_MR as usize][MATP_ND as usize] = 13;
    t.push(t1(&[0.006294030132, 0.015189408169, 0.258896467198, 0.015420910305, 0.449746529026, 0.053194553636])); // t[13]
    tsetmap[MATP_MR as usize][MATR_ND as usize] = 14;
    t.push(t1(&[0.020819322736, 0.000060497356, 0.272689176849, 0.063856784928])); // t[14]
    tsetmap[MATP_D as usize][BIF_ND as usize] = 15;
    t.push(t1(&[1.0, 1.0, 1.0])); // t[15]
    tsetmap[MATP_D as usize][END_ND as usize] = 16;
    t.push(t1(&[1.0, 1.0, 1.0])); // t[16]
    tsetmap[MATP_D as usize][MATL_ND as usize] = 17;
    t.push(t1(&[0.024577940691, 0.030655567559, 0.121290355765, 0.406621701238])); // t[17]
    tsetmap[MATP_D as usize][MATP_ND as usize] = 18;
    t.push(t1(&[0.001029025955, 0.002536729756, 0.046719556839, 0.029117903291, 0.028767509361, 0.436842892057])); // t[18]
    tsetmap[MATP_D as usize][MATR_ND as usize] = 19;
    t.push(t1(&[0.000017041108, 0.000007069171, 0.028384306256, 0.087965488640])); // t[19]
    tsetmap[MATP_IL as usize][BIF_ND as usize] = 20;
    t.push(t1(&[0.943443048986, 0.064001237265, 0.432230812455])); // t[20]
    tsetmap[MATP_IL as usize][END_ND as usize] = 21;
    t.push(t1(&[0.943443048986, 0.064001237265, 0.432230812455])); // t[21]
    tsetmap[MATP_IL as usize][MATL_ND as usize] = 22;
    t.push(t1(&[0.250101882938, 0.155728904821, 0.370945030932, 0.027811408475])); // t[22]
    tsetmap[MATP_IL as usize][MATP_ND as usize] = 23;
    t.push(t1(&[0.157307265492, 0.131105492208, 0.555106727689, 0.041624804903, 0.024305424386, 0.030756705205])); // t[23]
    tsetmap[MATP_IL as usize][MATR_ND as usize] = 24;
    t.push(t1(&[0.155093374292, 0.054734614999, 0.714409186001, 0.168407110635])); // t[24]
    tsetmap[MATP_IR as usize][BIF_ND as usize] = 25;
    t.push(t1(&[0.264643213319, 0.671462565227])); // t[25]
    tsetmap[MATP_IR as usize][END_ND as usize] = 26;
    t.push(t1(&[0.264643213319, 0.671462565227])); // t[26]
    tsetmap[MATP_IR as usize][MATL_ND as usize] = 27;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691])); // t[27]
    tsetmap[MATP_IR as usize][MATP_ND as usize] = 28;
    t.push(t1(&[0.291829430523, 1.098441427679, 0.025595408318, 0.091146313822, 0.042349119486])); // t[28]
    tsetmap[MATP_IR as usize][MATR_ND as usize] = 29;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204])); // t[29]
    tsetmap[MATL_ML as usize][BIF_ND as usize] = 30;
    t.push(t1(&[0.009635966745, 1.220143960207])); // t[30]
    tsetmap[MATL_ML as usize][END_ND as usize] = 31;
    t.push(t1(&[0.009635966745, 1.220143960207])); // t[31]
    tsetmap[MATL_ML as usize][MATL_ND as usize] = 32;
    t.push(t1(&[0.015185708311, 1.809432933023, 0.038601480352])); // t[32]
    tsetmap[MATL_ML as usize][MATP_ND as usize] = 33;
    t.push(t1(&[0.031820644019, 2.300193431878, 0.036163737927, 0.031218244200, 0.016826710214])); // t[33]
    tsetmap[MATL_ML as usize][MATR_ND as usize] = 34;
    t.push(t1(&[0.012395245929, 2.076134487839, 0.039781067793])); // t[34]
    tsetmap[MATL_D as usize][BIF_ND as usize] = 35;
    t.push(t1(&[0.019509171372, 6.781321301695])); // t[35]
    tsetmap[MATL_D as usize][END_ND as usize] = 36;
    t.push(t1(&[0.019509171372, 6.781321301695])); // t[36]
    tsetmap[MATL_D as usize][MATL_ND as usize] = 37;
    t.push(t1(&[0.005679808868, 0.127365862719, 0.277086556814])); // t[37]
    tsetmap[MATL_D as usize][MATP_ND as usize] = 38;
    t.push(t1(&[0.023424968753, 0.417640407951, 0.039088991906, 0.120577442402, 0.128103786646])); // t[38]
    tsetmap[MATL_D as usize][MATR_ND as usize] = 39;
    t.push(t1(&[0.013699691994, 0.405128575339, 0.254775565405])); // t[39]
    tsetmap[MATL_IL as usize][BIF_ND as usize] = 40;
    t.push(t1(&[0.264643213319, 0.671462565227])); // t[40]
    tsetmap[MATL_IL as usize][END_ND as usize] = 41;
    t.push(t1(&[0.264643213319, 0.671462565227])); // t[41]
    tsetmap[MATL_IL as usize][MATL_ND as usize] = 42;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691])); // t[42]
    tsetmap[MATL_IL as usize][MATP_ND as usize] = 43;
    t.push(t1(&[0.291829430523, 1.098441427679, 0.091146313822, 0.025595408318, 0.042349119486])); // t[43]
    tsetmap[MATL_IL as usize][MATR_ND as usize] = 44;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204])); // t[44]
    tsetmap[MATR_MR as usize][BIF_ND as usize] = 45;
    t.push(t1(&[0.009635966745, 1.220143960207])); // t[45]
    tsetmap[MATR_MR as usize][MATP_ND as usize] = 46;
    t.push(t1(&[0.031820644019, 2.300193431878, 0.036163737927, 0.031218244200, 0.016826710214])); // t[46]
    tsetmap[MATR_MR as usize][MATR_ND as usize] = 47;
    t.push(t1(&[0.012395245929, 2.076134487839, 0.039781067793])); // t[47]
    tsetmap[MATR_D as usize][BIF_ND as usize] = 48;
    t.push(t1(&[0.021604946951, 0.444765555211])); // t[48]
    tsetmap[MATR_D as usize][MATP_ND as usize] = 49;
    t.push(t1(&[0.021273745319, 0.532292228853, 0.110249350652, 0.040890357850, 0.164194410420])); // t[49]
    tsetmap[MATR_D as usize][MATR_ND as usize] = 50;
    t.push(t1(&[0.005806440507, 0.164264844267, 0.316876127883])); // t[50]
    tsetmap[MATR_IR as usize][BIF_ND as usize] = 51;
    t.push(t1(&[0.264643213319, 0.671462565227])); // t[51]
    tsetmap[MATR_IR as usize][MATP_ND as usize] = 52;
    t.push(t1(&[0.291829430523, 1.098441427679, 0.025595408318, 0.091146313822, 0.042349119486])); // t[52]
    tsetmap[MATR_IR as usize][MATR_ND as usize] = 53;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204])); // t[53]
    tsetmap[BEGL_S as usize][BIF_ND as usize] = 54;
    t.push(t1(&[1.0])); // t[54]
    tsetmap[BEGL_S as usize][MATP_ND as usize] = 55;
    t.push(t1(&[4.829712747509, 0.061131109227, 0.092185242101, 0.059154827887])); // t[55]
    tsetmap[BEGR_S as usize][BIF_ND as usize] = 56;
    t.push(t1(&[0.009635966745, 1.220143960207])); // t[56]
    tsetmap[BEGR_S as usize][MATL_ND as usize] = 57;
    t.push(t1(&[0.015185708311, 1.809432933023, 0.038601480352])); // t[57]
    tsetmap[BEGR_S as usize][MATP_ND as usize] = 58;
    t.push(t1(&[0.031820644019, 2.300193431878, 0.036163737927, 0.031218244200, 0.016826710214])); // t[58]
    tsetmap[BEGR_IL as usize][BIF_ND as usize] = 59;
    t.push(t1(&[0.264643213319, 0.671462565227])); // t[59]
    tsetmap[BEGR_IL as usize][MATL_ND as usize] = 60;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691])); // t[60]
    tsetmap[BEGR_IL as usize][MATP_ND as usize] = 61;
    t.push(t1(&[0.291829430523, 1.098441427679, 0.091146313822, 0.025595408318, 0.042349119486])); // t[61]
    tsetmap[ROOT_S as usize][BIF_ND as usize] = 62;
    t.push(t1(&[0.067710091654, 0.000047753225, 0.483183211040])); // t[62]
    tsetmap[ROOT_S as usize][MATL_ND as usize] = 63;
    t.push(t1(&[0.028518011579, 0.024705844026, 1.464047470747, 0.074164509948])); // t[63]
    tsetmap[ROOT_S as usize][MATP_ND as usize] = 64;
    t.push(t1(&[0.016729608598, 0.017449035307, 7.164604225972, 0.040744980202, 0.033562178957, 0.025523202345])); // t[64]
    tsetmap[ROOT_S as usize][MATR_ND as usize] = 65;
    t.push(t1(&[0.032901537296, 0.013876834787, 1.694917068307, 0.162141225286])); // t[65]
    tsetmap[ROOT_IL as usize][BIF_ND as usize] = 66;
    t.push(t1(&[0.943443048986, 0.064001237265, 0.432230812455])); // t[66]
    tsetmap[ROOT_IL as usize][MATL_ND as usize] = 67;
    t.push(t1(&[0.250101882938, 0.155728904821, 0.370945030932, 0.027811408475])); // t[67]
    tsetmap[ROOT_IL as usize][MATP_ND as usize] = 68;
    t.push(t1(&[0.157307265492, 0.131105492208, 0.555106727689, 0.041624804903, 0.024305424386, 0.030756705205])); // t[68]
    tsetmap[ROOT_IL as usize][MATR_ND as usize] = 69;
    t.push(t1(&[0.155093374292, 0.054734614999, 0.714409186001, 0.168407110635])); // t[69]
    tsetmap[ROOT_IR as usize][BIF_ND as usize] = 70;
    t.push(t1(&[0.264643213319, 0.671462565227])); // t[70]
    tsetmap[ROOT_IR as usize][MATL_ND as usize] = 71;
    t.push(t1(&[0.601223387577, 0.939499051719, 0.092516097691])); // t[71]
    tsetmap[ROOT_IR as usize][MATP_ND as usize] = 72;
    t.push(t1(&[0.291829430523, 1.098441427679, 0.025595408318, 0.091146313822, 0.042349119486])); // t[72]
    tsetmap[ROOT_IR as usize][MATR_ND as usize] = 73;
    t.push(t1(&[0.327208719748, 0.846283302435, 0.069337439204])); // t[73]
    let mut mbp = Mixdchlet::create(9, 16);
    mbp.q[0] = 0.030512242264;
    mbp.alpha[0] = vec![0.571860339721, 0.605642194896, 0.548004739487, 1.570353271532, 0.591611867703, 0.469713257214, 1.447411319683, 0.600381079228, 0.520096937350, 1.867142019076, 0.470428282443, 1.165356324744, 1.528348208160, 0.686072963473, 1.072148274499, 0.659833749087];
    mbp.q[1] = 0.070312169889;
    mbp.alpha[1] = vec![0.116757286812, 0.052661180881, 0.067541712113, 0.258482314714, 0.152527972588, 0.034460232010, 0.416430364713, 0.051541326273, 0.079542103337, 0.162883420833, 0.042615616796, 0.123363759874, 0.922897266376, 0.078567729294, 0.315242459757, 0.116457644231];
    mbp.q[2] = 0.118499696300;
    mbp.alpha[2] = vec![0.028961414077, 0.022849036260, 0.120089637379, 0.509884713979, 0.142464495045, 0.079507804767, 21.835608089779, 0.070200164694, 0.005189494879, 0.540651647339, 0.117833357497, 0.128182594376, 1.766866842025, 0.016341625779, 0.832665494899, 0.058379188171];
    mbp.q[3] = 0.181025557995;
    mbp.alpha[3] = vec![0.000926960236, 0.008100076237, 0.001794303710, 0.114209483231, 0.001459159085, 0.000053878201, 0.072605927746, 0.005533021345, 0.003941720307, 0.095421675098, 0.004844990769, 0.072393572779, 0.099144450569, 0.002561491533, 0.043103588084, 0.008080970629];
    mbp.q[4] = 0.188791659665;
    mbp.alpha[4] = vec![0.002163861165, 0.007785521817, 0.003483930554, 0.625515668281, 0.018621932500, 0.001352139642, 1.371471086809, 0.007920737783, 0.000946403264, 0.688821384972, 0.002203762108, 0.192533693864, 0.979473608513, 0.000916007398, 0.347662973488, 0.020677924150];
    mbp.q[5] = 0.157630937531;
    mbp.alpha[5] = vec![0.083035113547, 0.166815168558, 0.042669979127, 3.415107328082, 0.023530116520, 0.047677945396, 1.183956650707, 0.059920099115, 0.076614058723, 5.434261851985, 0.095284240991, 0.889915882997, 1.201576769946, 0.074453244946, 0.397879304331, 0.130525904952];
    mbp.q[6] = 0.041708924031;
    mbp.alpha[6] = vec![0.217001113139, 0.388746098242, 0.134680826556, 24.923110155367, 0.102582693868, 0.131678864943, 1.150978162882, 0.256720461728, 0.150993730345, 3.200824712363, 0.077595421397, 1.025428618792, 1.228870901327, 0.143610901605, 0.406308970402, 0.322809888354];
    mbp.q[7] = 0.095930656547;
    mbp.alpha[7] = vec![0.129043208355, 0.112308496092, 0.116841517642, 2.878927926806, 0.306789207829, 0.078411064993, 6.377836578660, 0.114524370807, 0.094192610036, 2.566493997218, 0.096694574300, 0.791295335090, 6.907854285192, 0.132657156809, 1.225349985791, 0.296596767798];
    mbp.q[8] = 0.115588155778;
    mbp.alpha[8] = vec![0.005830777296, 0.153807106950, 0.003131256711, 1.340589241710, 0.006802639527, 0.135277067812, 0.487492640368, 0.009160116179, 0.068942867388, 29.409376576276, 0.099733235653, 0.722700985558, 0.500134122079, 0.124671165331, 0.105694456385, 0.025741311658];
    let mut mnt = Mixdchlet::create(8, 4);
    mnt.q[0] = 0.085091850427;
    mnt.alpha[0] = vec![0.575686380127, 0.756214632926, 0.340269621276, 13.774558068728];
    mnt.q[1] = 0.015935406086;
    mnt.alpha[1] = vec![153.865583955384, 0.235000107300, 0.356622653787, 0.006812718667];
    mnt.q[2] = 0.102013232739;
    mnt.alpha[2] = vec![176.440373997567, 0.935905951648, 1.292808081312, 1.617069444109];
    mnt.q[3] = 0.415954530541;
    mnt.alpha[3] = vec![1.696250324914, 1.128033754503, 0.955462899400, 1.676465850057];
    mnt.q[4] = 0.074470557341;
    mnt.alpha[4] = vec![0.074365531036, 0.039185613484, 0.063868972113, 0.042432587902];
    mnt.q[5] = 0.055442639402;
    mnt.alpha[5] = vec![0.615068901818, 14.630712353118, 0.298404817403, 0.864718655041];
    mnt.q[6] = 0.118379098369;
    mnt.alpha[6] = vec![1.163176461349, 0.408090165233, 11.188793743319, 0.699118301558];
    mnt.q[7] = 0.132712685095;
    mnt.alpha[7] = vec![16.417200192194, 0.980503286582, 1.132071515554, 1.376129445524];
    let mut ii = Mixdchlet::create(8, 4);
    ii.q[0] = 0.085091850427;
    ii.alpha[0] = vec![0.575686380127, 0.756214632926, 0.340269621276, 13.774558068728];
    ii.q[1] = 0.015935406086;
    ii.alpha[1] = vec![153.865583955384, 0.235000107300, 0.356622653787, 0.006812718667];
    ii.q[2] = 0.102013232739;
    ii.alpha[2] = vec![176.440373997567, 0.935905951648, 1.292808081312, 1.617069444109];
    ii.q[3] = 0.415954530541;
    ii.alpha[3] = vec![1.696250324914, 1.128033754503, 0.955462899400, 1.676465850057];
    ii.q[4] = 0.074470557341;
    ii.alpha[4] = vec![0.074365531036, 0.039185613484, 0.063868972113, 0.042432587902];
    ii.q[5] = 0.055442639402;
    ii.alpha[5] = vec![0.615068901818, 14.630712353118, 0.298404817403, 0.864718655041];
    ii.q[6] = 0.118379098369;
    ii.alpha[6] = vec![1.163176461349, 0.408090165233, 11.188793743319, 0.699118301558];
    ii.q[7] = 0.132712685095;
    ii.alpha[7] = vec![16.417200192194, 0.980503286582, 1.132071515554, 1.376129445524];
    Prior {
        tsetnum: 74,
        tsetmap,
        t,
        mbp,
        mnt,
        i: ii,
        maxnq: 9,
        maxnalpha: 16,
    }
}
