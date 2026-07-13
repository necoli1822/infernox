//! Generic (non-SIMD) p7 profile DP used by Infernal's F4/F5 envelope
//! definition (`pli_p7_env_def`, cm_pipeline.c:2935). For the standard cmsearch
//! pass only the *glocal* generic path (`use_gm`) runs, so this module ports:
//!   - HMMER's nats-based `p7_FLogsum` lookup table (hmmer/src/logsum.c)
//!   - `p7_ProfileConfig(GLOCAL)` + `p7_ReconfigLength` (modelconfig.c:48-234)
//!   - `p7_GForward` / `p7_GBackward` (generic_fwdback.c)
//!
//! Every function embeds the actual C source in comments (see memory
//! `porting-comment-convention`). NOTE: this FLogsum is NATS with a lookup
//! table and is SEPARATE from Infernal's own bits-based logsum in cm_dp.rs.

use crate::p7_hmm::P7Profile;
use std::sync::OnceLock;

const NEG_INF: f32 = f32::NEG_INFINITY;

// ---------------------------------------------------------------------------
// c0: HMMER nats FLogsum lookup table (hmmer/src/logsum.c)
// ---------------------------------------------------------------------------

// C logsum.c:58-59:
//   #define p7_LOGSUM_SCALE 1000.f
//   #define p7_LOGSUM_TBL   16000
const P7_LOGSUM_SCALE: f32 = 1000.0;
const P7_LOGSUM_TBL: usize = 16000;

fn flogsum_table() -> &'static [f32; P7_LOGSUM_TBL] {
    // C p7_FLogsumInit (logsum.c):
    //   for (i = 0; i < p7_LOGSUM_TBL; i++)
    //     flogsum_lookup[i] = log(1. + exp((double) -i / p7_LOGSUM_SCALE));
    static TABLE: OnceLock<Box<[f32; P7_LOGSUM_TBL]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0.0f32; P7_LOGSUM_TBL]);
        for i in 0..P7_LOGSUM_TBL {
            t[i] = (1.0 + ((-(i as f64)) / P7_LOGSUM_SCALE as f64).exp()).ln() as f32;
        }
        t
    })
}

/// C p7_FLogsum (logsum.c:105):
///   const float max = ESL_MAX(a, b);
///   const float min = ESL_MIN(a, b);
///   return (min == -eslINFINITY || (max-min) >= 15.7f)
///          ? max : max + flogsum_lookup[(int)((max-min)*p7_LOGSUM_SCALE)];
#[inline]
pub fn p7_flogsum(a: f32, b: f32) -> f32 {
    let max = if a > b { a } else { b };
    let min = if a > b { b } else { a };
    if min == NEG_INF || (max - min) >= 15.7f32 {
        max
    } else {
        max + flogsum_table()[((max - min) * P7_LOGSUM_SCALE) as usize]
    }
}

// ---------------------------------------------------------------------------
// c1: glocal generic profile (p7_ProfileConfig GLOCAL + p7_ReconfigLength)
// ---------------------------------------------------------------------------

// p7P transition indices (hmmer.h:222-233, "order optimized for DP"):
const P7P_MM: usize = 0;
const P7P_IM: usize = 1;
const P7P_DM: usize = 2;
const P7P_BM: usize = 3;
const P7P_MD: usize = 4;
const P7P_DD: usize = 5;
const P7P_MI: usize = 6;
const P7P_II: usize = 7;
const P7P_NTRANS: usize = 8;

// p7H HMM-file transition indices, i.e. P7Profile.trans[k] = [MM,MI,MD,IM,II,DM,DD]
const P7H_MM: usize = 0;
const P7H_MI: usize = 1;
const P7H_MD: usize = 2;
const P7H_IM: usize = 3;
const P7H_II: usize = 4;
const P7H_DM: usize = 5;
const P7H_DD: usize = 6;

// xsc special-state row indices (hmmer.h): E=0,N=1,J=2,C=3; LOOP=0,MOVE=1.
const P7P_E: usize = 0;
const P7P_N: usize = 1;
const P7P_J: usize = 2;
const P7P_C: usize = 3;
const P7P_LOOP: usize = 0;
const P7P_MOVE: usize = 1;

/// Kp for RNA = 18, canonical K = 4.
const KP: usize = 18;
const K: usize = 4;

/// IUPAC RNA degeneracy table for sym "ACGU-RYMKSWHBVDN*~":
/// degen[x][i] = true if code x includes canonical residue i (A,C,G,U).
/// Used to fill degenerate match-emission scores (esl_abc_FExpectScVec).
const RNA_DEGEN: [[bool; 4]; KP] = [
    [true, false, false, false],  // 0 A
    [false, true, false, false],  // 1 C
    [false, false, true, false],  // 2 G
    [false, false, false, true],  // 3 U
    [false, false, false, false], // 4 - gap
    [true, false, true, false],   // 5 R = A,G
    [false, true, false, true],   // 6 Y = C,U
    [true, true, false, false],   // 7 M = A,C
    [false, false, true, true],   // 8 K = G,U
    [false, true, true, false],   // 9 S = C,G
    [true, false, false, true],   // 10 W = A,U
    [true, true, false, true],    // 11 H = A,C,U
    [false, true, true, true],    // 12 B = C,G,U
    [true, true, true, false],    // 13 V = A,C,G
    [true, false, true, true],    // 14 D = A,G,U
    [true, true, true, true],     // 15 N = A,C,G,U
    [false, false, false, false], // 16 * nonresidue
    [false, false, false, false], // 17 ~ missing
];

/// A glocal generic p7 profile: scores in NATS.
/// `tsc[k*8 + s]` (s in P7P_*), `rsc[x][k*2 + {0=MSC,1=ISC}]`, `xsc[state][LOOP|MOVE]`.
#[derive(Clone)]
pub struct GlocalProfile {
    pub m: usize,
    pub tsc: Vec<f32>,    // [(M+1)*8]
    pub rsc: Vec<Vec<f32>>, // [Kp][(M+1)*2]
    pub xsc: [[f32; 2]; 4],
    pub nj: f32,
    pub l: i32,
    /// C `p7_profile_IsLocal(gm->mode)`: TRUE for p7_LOCAL / p7_UNILOCAL (local
    /// alignment modes, ends emit-free from any node → generic Forward/Backward
    /// use `esc = 0`), FALSE for p7_GLOCAL / p7_UNIGLOCAL (`esc = -inf`, exit only
    /// from the last node). Default glocal profiles set this false; the truncated
    /// Lgm/Tgm profiles set it true. (generic_fwdback.c: `esc = IsLocal ? 0 : -inf`.)
    pub local_end: bool,
}

const P7P_NR: usize = 2;
const P7P_MSC: usize = 0;
const P7P_ISC: usize = 1;

/// Port of `p7_ProfileConfig(hmm, bg, gm, L, p7_GLOCAL)` (modelconfig.c:48-196)
/// followed by `p7_ReconfigLength(gm, L)`. Multihit glocal (default cmsearch
/// envelope-def profile), `bg->f[x] = 1/K = 0.25` for RNA.
pub fn build_glocal_profile(p7: &P7Profile, l: i32) -> GlocalProfile {
    let m = p7.m as usize;
    let bgf = 0.25f64; // bg->f[x] = 1/K for RNA
    let mut tsc = vec![0.0f32; (m + 1) * P7P_NTRANS];
    let mut rsc = vec![vec![0.0f32; (m + 1) * P7P_NR]; KP];

    // C 101-110 (glocal entry: left-wing retraction, log space):
    //   Z = log(hmm->t[0][p7H_MD]);
    //   p7P_TSC(gm,0,p7P_BM) = log(1.0 - hmm->t[0][p7H_MD]);
    //   for (k=1;k<hmm->M;k++) {
    //     p7P_TSC(gm,k,p7P_BM) = Z + log(hmm->t[k][p7H_DM]);
    //     Z += log(hmm->t[k][p7H_DD]); }
    let t0 = &p7.trans[0];
    // C modelconfig.c:103-108 uses `float Z` (truncated to f32 each accumulation
    // step); the `Z + log(...)` add promotes Z to double for the double log().
    // Match that: keep z in f32, do each add in f64 then round back to f32.
    let mut z: f32 = (t0[P7H_MD] as f64).ln() as f32;
    tsc[0 * P7P_NTRANS + P7P_BM] = (1.0 - t0[P7H_MD] as f64).ln() as f32;
    for k in 1..m {
        let tk = &p7.trans[k];
        tsc[k * P7P_NTRANS + P7P_BM] = (z as f64 + (tk[P7H_DM] as f64).ln()) as f32;
        z = (z as f64 + (tk[P7H_DD] as f64).ln()) as f32;
    }

    // C 125-135 (transition scores, k=1..M-1):
    //   tp[MM]=log t[k][MM]; tp[MI]=log t[k][MI]; tp[MD]=log t[k][MD];
    //   tp[IM]=log t[k][IM]; tp[II]=log t[k][II]; tp[DM]=log t[k][DM]; tp[DD]=log t[k][DD];
    for k in 1..m {
        let tk = &p7.trans[k];
        let base = k * P7P_NTRANS;
        tsc[base + P7P_MM] = (tk[P7H_MM] as f64).ln() as f32;
        tsc[base + P7P_MI] = (tk[P7H_MI] as f64).ln() as f32;
        tsc[base + P7P_MD] = (tk[P7H_MD] as f64).ln() as f32;
        tsc[base + P7P_IM] = (tk[P7H_IM] as f64).ln() as f32;
        tsc[base + P7P_II] = (tk[P7H_II] as f64).ln() as f32;
        tsc[base + P7P_DM] = (tk[P7H_DM] as f64).ln() as f32;
        tsc[base + P7P_DD] = (tk[P7H_DD] as f64).ln() as f32;
    }

    // C 137-151 (match emission scores):
    //   sc[K]=sc[Kp-2]=sc[Kp-1]=-inf;
    //   for (k=1;k<=M;k++){ for(x=0;x<K;x++) sc[x]=log(mat[k][x]/bg->f[x]);
    //     esl_abc_FExpectScVec(abc, sc, bg->f);
    //     for(x=0;x<Kp;x++) rsc[x][k*NR+MSC]=sc[x]; }
    for k in 1..=m {
        let mut sc = [0.0f32; KP];
        for x in 0..K {
            sc[x] = ((p7.mat[k][x] as f64) / bgf).ln() as f32;
        }
        sc[K] = NEG_INF; // gap
        sc[KP - 2] = NEG_INF; // nonresidue
        sc[KP - 1] = NEG_INF; // missing
        // esl_abc_FExpectScVec (esl_alphabet.c:1582): for x=K+1..Kp-3 fill
        //   sc[x] = sum_i degen[x][i]? sc[i]*p[i] : 0  / sum_i degen[x][i]? p[i] : 0
        for x in (K + 1)..=(KP - 3) {
            let mut result = 0.0f32;
            let mut denom = 0.0f32;
            for i in 0..K {
                if RNA_DEGEN[x][i] {
                    result += sc[i] * bgf as f32;
                    denom += bgf as f32;
                }
            }
            sc[x] = result / denom;
        }
        for x in 0..KP {
            rsc[x][k * P7P_NR + P7P_MSC] = sc[x];
        }
    }

    // C 153-169 (insert emission scores hardwired to 0, I_M impossible):
    //   for(x=0;x<Kp;x++){ for(k=1;k<M;k++) ISC(k,x)=0; ISC(M,x)=-inf; }
    //   ISC(k,K)=ISC(k,Kp-2)=ISC(k,Kp-1)=-inf for k=1..M
    for x in 0..KP {
        for k in 1..m {
            rsc[x][k * P7P_NR + P7P_ISC] = 0.0;
        }
        rsc[x][m * P7P_NR + P7P_ISC] = NEG_INF;
    }
    for k in 1..=m {
        rsc[K][k * P7P_NR + P7P_ISC] = NEG_INF;
        rsc[KP - 2][k * P7P_NR + P7P_ISC] = NEG_INF;
        rsc[KP - 1][k * P7P_NR + P7P_ISC] = NEG_INF;
    }

    // C 112-123 (E-state; multihit => loops allowed):
    //   xsc[E][MOVE]=xsc[E][LOOP]=-eslCONST_LOG2; nj=1.0
    let mut xsc = [[0.0f32; 2]; 4];
    let log2 = std::f64::consts::LN_2 as f32;
    xsc[P7P_E][P7P_MOVE] = -log2;
    xsc[P7P_E][P7P_LOOP] = -log2;
    let nj = 1.0f32;

    let mut gm = GlocalProfile { m, tsc, rsc, xsc, nj, l: 0, local_end: false };
    reconfig_length(&mut gm, l);
    gm
}

/// C `p7_ProfileConfig(hmm, bg, gm, L, p7_LOCAL)` followed by `p7_ReconfigLength`.
/// Identical to [`build_glocal_profile`] except the begin (BM) transitions use the
/// occupancy-weighted LOCAL entry distribution (modelconfig.c:96-108) and the
/// profile is in a local end mode (`esc = 0`). This is the base profile the
/// truncated `Tgm` (5'&3') is cloned from before `p7_ProfileConfig5PrimeAnd3PrimeTrunc`.
pub fn build_local_profile(p7: &P7Profile, l: i32) -> GlocalProfile {
    let mut gm = build_glocal_profile(p7, l);
    let m = gm.m;
    // Local mode entry (modelconfig.c:96-108, via p7_hmm_CalculateOccupancy):
    //   mocc[1] = t[0][MI] + t[0][MM]; mocc[k] = mocc[k-1]*(t[k-1][MM]+t[k-1][MI])
    //                                          + (1-mocc[k-1])*t[k-1][DM]
    //   Z = sum_{k=1..M} mocc[k]*(M-k+1); TSC(k-1,BM) = log(mocc[k]/Z)
    let mut mocc = vec![0.0f32; m + 1];
    mocc[1] = p7.trans[0][P7H_MI] + p7.trans[0][P7H_MM];
    for k in 2..=m {
        let tkm1 = &p7.trans[k - 1];
        mocc[k] = mocc[k - 1] * (tkm1[P7H_MM] + tkm1[P7H_MI])
            + (1.0 - mocc[k - 1]) * tkm1[P7H_DM];
    }
    let mut z = 0.0f32;
    for k in 1..=m {
        z += mocc[k] * (m - k + 1) as f32;
    }
    for k in 1..=m {
        // C modelconfig.c:98 log(occ[k] / Z): occ[k]/Z is float/float, log() is double.
        gm.tsc[(k - 1) * P7P_NTRANS + P7P_BM] = ((mocc[k] / z) as f64).ln() as f32;
    }
    gm.local_end = true; // p7_LOCAL is a local alignment mode
    gm
}

/// C `p7_ProfileConfig5PrimeTrunc(gm, L)` (cm_p7_modelconfig_trunc.c:30). Reconfigures
/// a GLOCAL profile `gm` in place for 5'-truncation (mode p7_UNIGLOCAL). Uniform
/// local begins log(1/M) into every node; N->N and E-loop made impossible (unihit,
/// exit only from node M — `esc` stays -inf because UNIGLOCAL is NOT a local mode).
pub fn p7_profile_config_5prime_trunc(gm: &mut GlocalProfile, l: i32) {
    let m = gm.m;
    // C cm_p7_modelconfig_trunc.c:41 log(1. / gm->M): 1. is double, M int → double.
    let inv = (1.0f64 / m as f64).ln() as f32;
    for k in 1..=m {
        gm.tsc[(k - 1) * P7P_NTRANS + P7P_BM] = inv;
    }
    // gm->mode = p7_UNIGLOCAL: still non-local → esc = -inf (leave local_end=false).
    gm.local_end = false;
    // ReconfigUnihit-ish: J unreachable, N->N impossible.
    gm.xsc[P7P_N][P7P_MOVE] = 0.0;
    gm.xsc[P7P_E][P7P_MOVE] = 0.0;
    gm.xsc[P7P_N][P7P_LOOP] = NEG_INF;
    gm.xsc[P7P_E][P7P_LOOP] = NEG_INF;
    gm.l = 0;
    p7_reconfig_length_5prime_trunc(gm, l);
}

/// C `p7_ProfileConfig3PrimeTrunc(hmm, gm, L)` (cm_p7_modelconfig_trunc.c:65).
/// Reconfigures a GLOCAL profile `gm` in place for 3'-truncation (mode p7_UNILOCAL).
/// Glocal left-wing-retracted begins (recomputed here), C->C & E-loop impossible;
/// mode is a LOCAL alignment mode so `esc = 0` (exit allowed from any node).
pub fn p7_profile_config_3prime_trunc(p7: &P7Profile, gm: &mut GlocalProfile, l: i32) {
    let m = gm.m;
    // glocal-mode entry: left wing retraction, log space (identical to glocal build).
    let t0 = &p7.trans[0];
    // C uses `float Z` (truncated to f32 each step); add promotes to double for log().
    let mut z: f32 = (t0[P7H_MD] as f64).ln() as f32;
    gm.tsc[0 * P7P_NTRANS + P7P_BM] = (1.0 - t0[P7H_MD] as f64).ln() as f32;
    for k in 1..m {
        let tk = &p7.trans[k];
        gm.tsc[k * P7P_NTRANS + P7P_BM] = (z as f64 + (tk[P7H_DM] as f64).ln()) as f32;
        z = (z as f64 + (tk[P7H_DD] as f64).ln()) as f32;
    }
    // gm->mode = p7_UNILOCAL → local end scores allowed (esc = 0).
    gm.local_end = true;
    gm.xsc[P7P_C][P7P_MOVE] = 0.0;
    gm.xsc[P7P_E][P7P_MOVE] = 0.0;
    gm.xsc[P7P_C][P7P_LOOP] = NEG_INF;
    gm.xsc[P7P_E][P7P_LOOP] = NEG_INF;
    gm.l = 0;
    p7_reconfig_length_3prime_trunc(gm, l);
}

/// C `p7_ProfileConfig5PrimeAnd3PrimeTrunc(gm, L)` (cm_p7_modelconfig_trunc.c:106).
/// `gm` must already be a LOCAL profile (equiprobable begins/ends). Make it unihit:
/// N->N, C->C, E-loop impossible; nj=0; mode p7_UNILOCAL. Does NOT call ReconfigLength.
pub fn p7_profile_config_5prime_and_3prime_trunc(gm: &mut GlocalProfile, l: i32) {
    gm.xsc[P7P_N][P7P_MOVE] = 0.0;
    gm.xsc[P7P_C][P7P_MOVE] = 0.0;
    gm.xsc[P7P_E][P7P_MOVE] = 0.0;
    gm.xsc[P7P_N][P7P_LOOP] = NEG_INF;
    gm.xsc[P7P_C][P7P_LOOP] = NEG_INF;
    gm.xsc[P7P_E][P7P_LOOP] = NEG_INF;
    gm.nj = 0.0;
    gm.l = l;
    gm.local_end = true; // p7_UNILOCAL
}

/// C `p7_ReconfigLength5PrimeTrunc(gm, L)` (cm_p7_modelconfig_trunc.c:124). C-loop/move
/// bear the sequence length; pmove = (1+nj)/(L+1+nj).
pub fn p7_reconfig_length_5prime_trunc(gm: &mut GlocalProfile, l: i32) {
    let pmove = (1.0f32 + gm.nj) / (l as f32 + 1.0f32 + gm.nj);
    let ploop = 1.0f32 - pmove;
    let ll = (ploop as f64).ln() as f32;
    let lm = (pmove as f64).ln() as f32;
    gm.xsc[P7P_C][P7P_LOOP] = ll;
    gm.xsc[P7P_J][P7P_LOOP] = ll;
    gm.xsc[P7P_C][P7P_MOVE] = lm;
    gm.xsc[P7P_J][P7P_MOVE] = lm;
    gm.l = l;
}

/// C `p7_ReconfigLength3PrimeTrunc(gm, L)` (cm_p7_modelconfig_trunc.c:149). N-loop/move
/// bear the sequence length; pmove = (1+nj)/(L+1+nj).
pub fn p7_reconfig_length_3prime_trunc(gm: &mut GlocalProfile, l: i32) {
    let pmove = (1.0f32 + gm.nj) / (l as f32 + 1.0f32 + gm.nj);
    let ploop = 1.0f32 - pmove;
    let ll = (ploop as f64).ln() as f32;
    let lm = (pmove as f64).ln() as f32;
    gm.xsc[P7P_N][P7P_LOOP] = ll;
    gm.xsc[P7P_J][P7P_LOOP] = ll;
    gm.xsc[P7P_N][P7P_MOVE] = lm;
    gm.xsc[P7P_J][P7P_MOVE] = lm;
    gm.l = l;
}

/// Port of `p7_ReconfigLength(gm, L)` (modelconfig.c:220-234).
///   pmove = (2 + nj) / (L + 2 + nj); ploop = 1 - pmove;
///   xsc[N/C/J][LOOP]=log(ploop); xsc[N/C/J][MOVE]=log(pmove);
pub fn reconfig_length(gm: &mut GlocalProfile, l: i32) {
    let pmove = (2.0f32 + gm.nj) / (l as f32 + 2.0f32 + gm.nj);
    let ploop = 1.0f32 - pmove;
    // C modelconfig.c: log(ploop) / log(pmove) — args are float, log() is double.
    let ll = (ploop as f64).ln() as f32;
    let lm = (pmove as f64).ln() as f32;
    for &s in &[P7P_N, P7P_C, P7P_J] {
        gm.xsc[s][P7P_LOOP] = ll;
        gm.xsc[s][P7P_MOVE] = lm;
    }
    gm.l = l;
}

// ---------------------------------------------------------------------------
// Generic DP matrix (p7_gmx) — dp[i][k*3 + {M,I,D}], xmx[i][E,N,J,B,C]
// ---------------------------------------------------------------------------

const P7G_M: usize = 0;
const P7G_I: usize = 1;
const P7G_D: usize = 2;
const P7G_NSCELLS: usize = 3;
const P7G_E: usize = 0;
const P7G_N: usize = 1;
const P7G_J: usize = 2;
const P7G_B: usize = 3;
const P7G_C: usize = 4;
const P7G_NXCELLS: usize = 5;

pub struct P7Gmx {
    pub m: usize,
    pub l: usize,
    pub dp: Vec<f32>,  // [(L+1)*(M+1)*3]
    pub xmx: Vec<f32>, // [(L+1)*5]
}

impl P7Gmx {
    pub fn new(m: usize, l: usize) -> Self {
        P7Gmx {
            m,
            l,
            dp: vec![0.0; (l + 1) * (m + 1) * P7G_NSCELLS],
            xmx: vec![0.0; (l + 1) * P7G_NXCELLS],
        }
    }
    // The DP matrix is sized (l+1)*(m+1)*NSCELLS at construction and every caller
    // indexes within i<=l, k<=m (byte-verified vs C). Elide the per-access bounds
    // check; a debug_assert keeps the safety net in debug builds.
    //
    // LAYOUT: planar-WITHIN-row. Row i occupies the contiguous block
    //   dp[i*(m+1)*3 .. (i+1)*(m+1)*3]  =  [ M[0..=m] | I[0..=m] | D[0..=m] ]
    // so M/I/D values for a single row are each CONTIGUOUS (stride 1) over k —
    // required by the AVX2 fast path in `p7_gforward` (contiguous loads/stores over
    // the k dimension). The per-row block is still exactly (m+1)*3 wide, so the
    // row-elementwise summing in `p7_gnull2_by_expectation` (which treats each row
    // as one contiguous block) stays correct: cell c of every row maps to the same
    // (plane,k). This is a pure reindex of the SAME scalar values — bit-identical.
    #[inline]
    fn plane_base(&self, i: usize, plane: usize) -> usize {
        i * (self.m + 1) * P7G_NSCELLS + plane * (self.m + 1)
    }
    #[inline]
    fn mmx(&self, i: usize, k: usize) -> f32 {
        let idx = self.plane_base(i, P7G_M) + k;
        debug_assert!(idx < self.dp.len());
        unsafe { *self.dp.get_unchecked(idx) }
    }
    #[inline]
    fn imx(&self, i: usize, k: usize) -> f32 {
        let idx = self.plane_base(i, P7G_I) + k;
        debug_assert!(idx < self.dp.len());
        unsafe { *self.dp.get_unchecked(idx) }
    }
    #[inline]
    fn dmx(&self, i: usize, k: usize) -> f32 {
        let idx = self.plane_base(i, P7G_D) + k;
        debug_assert!(idx < self.dp.len());
        unsafe { *self.dp.get_unchecked(idx) }
    }
    #[inline]
    fn xmx(&self, i: usize, s: usize) -> f32 {
        let idx = i * P7G_NXCELLS + s;
        debug_assert!(idx < self.xmx.len());
        unsafe { *self.xmx.get_unchecked(idx) }
    }
    #[inline]
    fn set_mmx(&mut self, i: usize, k: usize, v: f32) {
        let idx = self.plane_base(i, P7G_M) + k;
        debug_assert!(idx < self.dp.len());
        unsafe { *self.dp.get_unchecked_mut(idx) = v; }
    }
    #[inline]
    fn set_imx(&mut self, i: usize, k: usize, v: f32) {
        let idx = self.plane_base(i, P7G_I) + k;
        debug_assert!(idx < self.dp.len());
        unsafe { *self.dp.get_unchecked_mut(idx) = v; }
    }
    #[inline]
    fn set_dmx(&mut self, i: usize, k: usize, v: f32) {
        let idx = self.plane_base(i, P7G_D) + k;
        debug_assert!(idx < self.dp.len());
        unsafe { *self.dp.get_unchecked_mut(idx) = v; }
    }
    #[inline]
    fn set_xmx(&mut self, i: usize, s: usize, v: f32) {
        let idx = i * P7G_NXCELLS + s;
        debug_assert!(idx < self.xmx.len());
        unsafe { *self.xmx.get_unchecked_mut(idx) = v; }
    }
}

#[inline]
fn tsc(gm: &GlocalProfile, s: usize, k: usize) -> f32 {
    gm.tsc[k * P7P_NTRANS + s]
}

// ---------------------------------------------------------------------------
// c2: p7_GForward (generic_fwdback.c:47-137), glocal (esc = -inf)
// ---------------------------------------------------------------------------

/// Runtime dispatch flags for the gforward fast path, resolved once.
/// `INFERNOX_GFWD_SCALAR=1` forces the scalar reference path (trivial revert /
/// A-B benchmarking). `IX_GFWD_VERIFY=1` runs BOTH paths on every call and asserts
/// the full DP matrix + score are f32 bit-identical (kernel-level parity proof).
#[cfg(target_arch = "x86_64")]
fn gfwd_force_scalar() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("INFERNOX_GFWD_SCALAR").is_some())
}
#[cfg(target_arch = "x86_64")]
fn gfwd_verify() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("IX_GFWD_VERIFY").is_some())
}

/// Port of `p7_GForward`. `dsq` is 1-indexed with sentinels; residues at 1..=l.
/// Returns the Forward lod score in NATS and fills the DP matrix (for Backward
/// and posterior decoding). Glocal mode: `esc = -eslINFINITY`. Dispatches to a
/// bit-exact AVX2 fast path when available.
pub fn p7_gforward(dsq: &[u8], l: usize, gm: &GlocalProfile, gx: &mut P7Gmx) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && !gfwd_force_scalar() {
            if gfwd_verify() {
                // Kernel-level bit-exact proof: run the scalar reference into a
                // scratch matrix, the SIMD path into `gx`, assert exact equality of
                // the full DP core, all xmx cells, and the returned score.
                let mut gref = P7Gmx::new(gm.m, l);
                let s_ref = p7_gforward_scalar(dsq, l, gm, &mut gref);
                let s_simd = unsafe { p7_gforward_avx2(dsq, l, gm, gx) };
                for (idx, (a, b)) in gref.dp.iter().zip(gx.dp.iter()).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "gforward VERIFY dp[{idx}] scalar={a} simd={b} (m={}, l={})",
                        gm.m,
                        l
                    );
                }
                for (idx, (a, b)) in gref.xmx.iter().zip(gx.xmx.iter()).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "gforward VERIFY xmx[{idx}] scalar={a} simd={b}"
                    );
                }
                assert!(
                    s_ref.to_bits() == s_simd.to_bits(),
                    "gforward VERIFY score scalar={s_ref} simd={s_simd}"
                );
                return s_simd;
            }
            return unsafe { p7_gforward_avx2(dsq, l, gm, gx) };
        }
    }
    p7_gforward_scalar(dsq, l, gm, gx)
}

/// Scalar reference transcription of `p7_GForward` (generic_fwdback.c:47-137).
/// The authoritative byte-parity implementation; the AVX2 path must reproduce its
/// output f32-exactly (see `IX_GFWD_VERIFY`).
pub fn p7_gforward_scalar(dsq: &[u8], l: usize, gm: &GlocalProfile, gx: &mut P7Gmx) -> f32 {
    let mm = gm.m;
    let esc = if gm.local_end { 0.0 } else { NEG_INF }; // p7_profile_IsLocal(gm) ? 0 : -eslINFINITY

    // C 59-64: zero row.
    //   XMX(0,N)=0; XMX(0,B)=xsc[N][MOVE]; XMX(0,E)=XMX(0,C)=XMX(0,J)=-inf;
    //   for k=0..M: MMX(0,k)=IMX(0,k)=DMX(0,k)=-inf;
    gx.set_xmx(0, P7G_N, 0.0);
    gx.set_xmx(0, P7G_B, gm.xsc[P7P_N][P7P_MOVE]);
    gx.set_xmx(0, P7G_E, NEG_INF);
    gx.set_xmx(0, P7G_C, NEG_INF);
    gx.set_xmx(0, P7G_J, NEG_INF);
    for k in 0..=mm {
        gx.set_mmx(0, k, NEG_INF);
        gx.set_imx(0, k, NEG_INF);
        gx.set_dmx(0, k, NEG_INF);
    }

    // C 71-131: recursion.
    for i in 1..=l {
        let x = dsq[i] as usize;
        // rsc = gm->rsc[dsq[i]]; MSC(k)=rsc[k*2+MSC]; ISC(k)=rsc[k*2+ISC]
        let rscx = &gm.rsc[x];

        gx.set_mmx(i, 0, NEG_INF);
        gx.set_imx(i, 0, NEG_INF);
        gx.set_dmx(i, 0, NEG_INF);
        gx.set_xmx(i, P7G_E, NEG_INF);

        for k in 1..mm {
            // match state (C 82-86)
            let sc = p7_flogsum(
                p7_flogsum(
                    gx.mmx(i - 1, k - 1) + tsc(gm, P7P_MM, k - 1),
                    gx.imx(i - 1, k - 1) + tsc(gm, P7P_IM, k - 1),
                ),
                p7_flogsum(
                    gx.xmx(i - 1, P7G_B) + tsc(gm, P7P_BM, k - 1),
                    gx.dmx(i - 1, k - 1) + tsc(gm, P7P_DM, k - 1),
                ),
            );
            gx.set_mmx(i, k, sc + rscx[k * P7P_NR + P7P_MSC]);

            // insert state (C 89-91)
            let sc = p7_flogsum(
                gx.mmx(i - 1, k) + tsc(gm, P7P_MI, k),
                gx.imx(i - 1, k) + tsc(gm, P7P_II, k),
            );
            gx.set_imx(i, k, sc + rscx[k * P7P_NR + P7P_ISC]);

            // delete state (C 94-95)
            let d = p7_flogsum(
                gx.mmx(i, k - 1) + tsc(gm, P7P_MD, k - 1),
                gx.dmx(i, k - 1) + tsc(gm, P7P_DD, k - 1),
            );
            gx.set_dmx(i, k, d);

            // E state update (C 98-100)
            let e = p7_flogsum(
                p7_flogsum(gx.mmx(i, k) + esc, gx.dmx(i, k) + esc),
                gx.xmx(i, P7G_E),
            );
            gx.set_xmx(i, P7G_E, e);
        }

        // unrolled M_M (C 103-107)
        let sc = p7_flogsum(
            p7_flogsum(
                gx.mmx(i - 1, mm - 1) + tsc(gm, P7P_MM, mm - 1),
                gx.imx(i - 1, mm - 1) + tsc(gm, P7P_IM, mm - 1),
            ),
            p7_flogsum(
                gx.xmx(i - 1, P7G_B) + tsc(gm, P7P_BM, mm - 1),
                gx.dmx(i - 1, mm - 1) + tsc(gm, P7P_DM, mm - 1),
            ),
        );
        gx.set_mmx(i, mm, sc + rscx[mm * P7P_NR + P7P_MSC]);
        gx.set_imx(i, mm, NEG_INF);

        // unrolled D_M (C 111-112)
        let d = p7_flogsum(
            gx.mmx(i, mm - 1) + tsc(gm, P7P_MD, mm - 1),
            gx.dmx(i, mm - 1) + tsc(gm, P7P_DD, mm - 1),
        );
        gx.set_dmx(i, mm, d);

        // unrolled E update (C 115-117)
        let e = p7_flogsum(p7_flogsum(gx.mmx(i, mm), gx.dmx(i, mm)), gx.xmx(i, P7G_E));
        gx.set_xmx(i, P7G_E, e);

        // J state (C 120-121)
        let j = p7_flogsum(
            gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
            gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP],
        );
        gx.set_xmx(i, P7G_J, j);

        // C state (C 123-124)
        let c = p7_flogsum(
            gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP],
            gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_C, c);

        // N state (C 126)
        let n = gx.xmx(i - 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP];
        gx.set_xmx(i, P7G_N, n);

        // B state (C 129-130)
        let b = p7_flogsum(
            gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE],
            gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_B, b);
    }

    // C 133: *opt_sc = XMX(L,C) + xsc[C][MOVE]
    gx.m = mm;
    gx.l = l;
    gx.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE]
}

// ---------------------------------------------------------------------------
// c2b: AVX2 fast path for p7_GForward.
//
// STRUCTURE ANALYSIS (per DP row i, generic_fwdback.c inner loop k=1..M):
//   M[i,k] = flogsum( flogsum(M[i-1,k-1]+MM, I[i-1,k-1]+IM),
//                     flogsum(B[i-1]+BM,     D[i-1,k-1]+DM) ) + MSC(k)
//   I[i,k] = flogsum( M[i-1,k]+MI, I[i-1,k]+II ) + ISC(k)
//   D[i,k] = flogsum( M[i,k-1]+MD, D[i,k-1]+DD )              <-- k-1 serial
//   E[i]  += flogsum( M[i,k]+esc, D[i,k]+esc )                <-- serial reduction
//
//   * M[i,k] and I[i,k] depend ONLY on row i-1 (and the scalar B[i-1]) — they are
//     INDEPENDENT across k, so vectorizing over the k dimension keeps each cell's
//     exact operand set and its exact per-cell flogsum accumulation order. That is
//     BIT-EXACT to the scalar recurrence (same reason as the task-#14 CYK fix: lanes
//     over independent cells, never over a reduction).
//   * D[i,k] carries a k-1 dependency within the row → kept SCALAR (serial scan).
//   * E[i] is a left-to-right flogsum reduction (non-associative) → kept SCALAR in
//     the SAME k order as C. In glocal (esc = -inf) the in-loop E update is a proven
//     identity (flogsum(-inf,-inf)=-inf; flogsum(-inf,E)=E) and is skipped.
//
// Vectorized: the M and I planes (the bulk of the flogsum work). Serial: D and E.
// The layout is planar-within-row so M[i-1,*], I[i-1,*], D[i-1,*] load contiguously
// and M[i,*]/I[i,*] store contiguously.
// ---------------------------------------------------------------------------

/// AVX2 8-wide `p7_flogsum` (bit-exact to scalar `p7_flogsum` lane-for-lane).
/// `max=max(a,b); min=min(a,b); diff=max-min;` result = (min==-inf || diff>=15.7)
/// ? max : max + table[(i32)(diff*1000)]. Masked-out lanes gather index 0 (safe,
/// blended away) so a +inf/NaN diff can never produce an out-of-bounds gather.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn flogsum8(a: std::arch::x86_64::__m256, b: std::arch::x86_64::__m256, tptr: *const f32)
    -> std::arch::x86_64::__m256
{
    use std::arch::x86_64::*;
    let max = _mm256_max_ps(a, b);
    let min = _mm256_min_ps(a, b);
    let diff = _mm256_sub_ps(max, min); // >=0 for finite; NaN if a=b=-inf
    let neginf = _mm256_set1_ps(NEG_INF);
    // cond = (min == -inf) | (diff >= 15.7)   [NaN diff -> GE is false; min==-inf catches it]
    let is_min_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(min, neginf);
    let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(diff, _mm256_set1_ps(15.7));
    let cond = _mm256_or_ps(is_min_inf, ge);
    // idx = (i32) (diff * 1000.0)   (truncation toward zero == C's (int) cast)
    let scaled = _mm256_mul_ps(diff, _mm256_set1_ps(P7_LOGSUM_SCALE));
    let idx = _mm256_cvttps_epi32(scaled);
    // safe_idx = cond ? 0 : idx  (avoid OOB gather where cond is true)
    let condi = _mm256_castps_si256(cond);
    let safe_idx = _mm256_andnot_si256(condi, idx);
    let tbl = _mm256_i32gather_ps::<4>(tptr, safe_idx);
    let sum = _mm256_add_ps(max, tbl);
    // result = cond ? max : sum
    _mm256_blendv_ps(sum, max, cond)
}

/// AVX2 fast path for [`p7_gforward`]. Bit-exact to `p7_gforward_scalar` (proven per
/// call under `IX_GFWD_VERIFY`). Vectorizes the M/I row updates over k; D and E stay
/// scalar/serial in exact C order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn p7_gforward_avx2(dsq: &[u8], l: usize, gm: &GlocalProfile, gx: &mut P7Gmx) -> f32 {
    use std::arch::x86_64::*;
    let mm = gm.m;
    let esc = if gm.local_end { 0.0f32 } else { NEG_INF };
    let local = esc == 0.0f32; // esc is exactly 0.0 (local) or -inf (glocal)
    let table = flogsum_table();
    let tptr = table.as_ptr();
    let m1 = mm + 1;
    let rowsz = m1 * P7G_NSCELLS;

    // --- transpose transition scores used by the vectorized M/I updates ---
    // t_*[k] = gm.tsc[k*P7P_NTRANS + s]  (contiguous over k).
    let mut t_mm = vec![NEG_INF; m1];
    let mut t_im = vec![NEG_INF; m1];
    let mut t_bm = vec![NEG_INF; m1];
    let mut t_dm = vec![NEG_INF; m1];
    let mut t_mi = vec![NEG_INF; m1];
    let mut t_ii = vec![NEG_INF; m1];
    for k in 0..m1 {
        let base = k * P7P_NTRANS;
        t_mm[k] = gm.tsc[base + P7P_MM];
        t_im[k] = gm.tsc[base + P7P_IM];
        t_bm[k] = gm.tsc[base + P7P_BM];
        t_dm[k] = gm.tsc[base + P7P_DM];
        t_mi[k] = gm.tsc[base + P7P_MI];
        t_ii[k] = gm.tsc[base + P7P_II];
    }
    // --- transpose emission scores per symbol: msc_t[x*m1+k], isc_t[x*m1+k] ---
    let kp = gm.rsc.len();
    let mut msc_t = vec![NEG_INF; kp * m1];
    let mut isc_t = vec![NEG_INF; kp * m1];
    for x in 0..kp {
        let r = &gm.rsc[x];
        let off = x * m1;
        for k in 0..m1 {
            msc_t[off + k] = r[k * P7P_NR + P7P_MSC];
            isc_t[off + k] = r[k * P7P_NR + P7P_ISC];
        }
    }

    // C 59-64: zero row.
    gx.set_xmx(0, P7G_N, 0.0);
    gx.set_xmx(0, P7G_B, gm.xsc[P7P_N][P7P_MOVE]);
    gx.set_xmx(0, P7G_E, NEG_INF);
    gx.set_xmx(0, P7G_C, NEG_INF);
    gx.set_xmx(0, P7G_J, NEG_INF);
    for k in 0..=mm {
        gx.set_mmx(0, k, NEG_INF);
        gx.set_imx(0, k, NEG_INF);
        gx.set_dmx(0, k, NEG_INF);
    }

    for i in 1..=l {
        let x = dsq[i] as usize;
        let mscx = &msc_t[x * m1..x * m1 + m1];
        let iscx = &isc_t[x * m1..x * m1 + m1];

        // C: MMX(i,0)=IMX(i,0)=DMX(i,0)=-inf; XMX(i,E)=-inf.
        gx.set_mmx(i, 0, NEG_INF);
        gx.set_imx(i, 0, NEG_INF);
        gx.set_dmx(i, 0, NEG_INF);
        gx.set_xmx(i, P7G_E, NEG_INF);

        let bprev = gx.xmx(i - 1, P7G_B);

        // --- vectorized M/I updates over k in [1, mm-1] ---
        {
            let row_cur = i * rowsz;
            let row_prev = (i - 1) * rowsz;
            let (left, right) = gx.dp.split_at_mut(row_cur);
            let prev_m = left.as_ptr().add(row_prev + P7G_M * m1);
            let prev_i = left.as_ptr().add(row_prev + P7G_I * m1);
            let prev_d = left.as_ptr().add(row_prev + P7G_D * m1);
            let cur_m = right.as_mut_ptr().add(P7G_M * m1);
            let cur_i = right.as_mut_ptr().add(P7G_I * m1);

            let bv = _mm256_set1_ps(bprev);
            let mut k = 1usize;
            // SIMD blocks: process k..k+7 while k+8 <= mm (so k+7 <= mm-1).
            while k + 8 <= mm {
                // M[i,k] = flogsum(flogsum(M[i-1,k-1]+MM, I[i-1,k-1]+IM),
                //                  flogsum(B[i-1]+BM,     D[i-1,k-1]+DM)) + MSC(k)
                let km1 = k - 1;
                let va = _mm256_add_ps(_mm256_loadu_ps(prev_m.add(km1)),
                                       _mm256_loadu_ps(t_mm.as_ptr().add(km1)));
                let vb = _mm256_add_ps(_mm256_loadu_ps(prev_i.add(km1)),
                                       _mm256_loadu_ps(t_im.as_ptr().add(km1)));
                let vc = _mm256_add_ps(bv, _mm256_loadu_ps(t_bm.as_ptr().add(km1)));
                let vd = _mm256_add_ps(_mm256_loadu_ps(prev_d.add(km1)),
                                       _mm256_loadu_ps(t_dm.as_ptr().add(km1)));
                let sc = flogsum8(flogsum8(va, vb, tptr), flogsum8(vc, vd, tptr), tptr);
                let vm = _mm256_add_ps(sc, _mm256_loadu_ps(mscx.as_ptr().add(k)));
                _mm256_storeu_ps(cur_m.add(k), vm);

                // I[i,k] = flogsum(M[i-1,k]+MI, I[i-1,k]+II) + ISC(k)
                let vp = _mm256_add_ps(_mm256_loadu_ps(prev_m.add(k)),
                                       _mm256_loadu_ps(t_mi.as_ptr().add(k)));
                let vq = _mm256_add_ps(_mm256_loadu_ps(prev_i.add(k)),
                                       _mm256_loadu_ps(t_ii.as_ptr().add(k)));
                let si = flogsum8(vp, vq, tptr);
                let vi = _mm256_add_ps(si, _mm256_loadu_ps(iscx.as_ptr().add(k)));
                _mm256_storeu_ps(cur_i.add(k), vi);

                k += 8;
            }
            // scalar remainder for k in [k, mm-1]
            while k < mm {
                let km1 = k - 1;
                let scm = p7_flogsum(
                    p7_flogsum(*prev_m.add(km1) + t_mm[km1], *prev_i.add(km1) + t_im[km1]),
                    p7_flogsum(bprev + t_bm[km1], *prev_d.add(km1) + t_dm[km1]),
                );
                *cur_m.add(k) = scm + mscx[k];
                let sci = p7_flogsum(*prev_m.add(k) + t_mi[k], *prev_i.add(k) + t_ii[k]);
                *cur_i.add(k) = sci + iscx[k];
                k += 1;
            }
        }

        // --- serial D scan (+ E reduction in local mode), exact C k-order ---
        for k in 1..mm {
            let d = p7_flogsum(
                gx.mmx(i, k - 1) + tsc(gm, P7P_MD, k - 1),
                gx.dmx(i, k - 1) + tsc(gm, P7P_DD, k - 1),
            );
            gx.set_dmx(i, k, d);
            if local {
                let e = p7_flogsum(
                    p7_flogsum(gx.mmx(i, k) + esc, gx.dmx(i, k) + esc),
                    gx.xmx(i, P7G_E),
                );
                gx.set_xmx(i, P7G_E, e);
            }
        }

        // unrolled M_M (C 103-107)
        let sc = p7_flogsum(
            p7_flogsum(
                gx.mmx(i - 1, mm - 1) + tsc(gm, P7P_MM, mm - 1),
                gx.imx(i - 1, mm - 1) + tsc(gm, P7P_IM, mm - 1),
            ),
            p7_flogsum(
                gx.xmx(i - 1, P7G_B) + tsc(gm, P7P_BM, mm - 1),
                gx.dmx(i - 1, mm - 1) + tsc(gm, P7P_DM, mm - 1),
            ),
        );
        gx.set_mmx(i, mm, sc + mscx[mm]);
        gx.set_imx(i, mm, NEG_INF);

        // unrolled D_M (C 111-112)
        let d = p7_flogsum(
            gx.mmx(i, mm - 1) + tsc(gm, P7P_MD, mm - 1),
            gx.dmx(i, mm - 1) + tsc(gm, P7P_DD, mm - 1),
        );
        gx.set_dmx(i, mm, d);

        // unrolled E update (C 115-117)
        let e = p7_flogsum(p7_flogsum(gx.mmx(i, mm), gx.dmx(i, mm)), gx.xmx(i, P7G_E));
        gx.set_xmx(i, P7G_E, e);

        // J state (C 120-121)
        let j = p7_flogsum(
            gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
            gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP],
        );
        gx.set_xmx(i, P7G_J, j);

        // C state (C 123-124)
        let c = p7_flogsum(
            gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP],
            gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_C, c);

        // N state (C 126)
        let n = gx.xmx(i - 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP];
        gx.set_xmx(i, P7G_N, n);

        // B state (C 129-130)
        let b = p7_flogsum(
            gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE],
            gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_B, b);
    }

    gx.m = mm;
    gx.l = l;
    gx.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE]
}

// ---------------------------------------------------------------------------
// c4: p7_GBackward (generic_fwdback.c:163-230), glocal (esc = -inf)
// ---------------------------------------------------------------------------

/// Port of `p7_GBackward`. Fills `gx` with the Backward matrix and returns the
/// Backward lod score in NATS.
pub fn p7_gbackward(dsq: &[u8], l: usize, gm: &GlocalProfile, gx: &mut P7Gmx) -> f32 {
    let mm = gm.m;
    let esc = if gm.local_end { 0.0 } else { NEG_INF };

    // C 180-192: initialize the L row.
    gx.set_xmx(l, P7G_J, NEG_INF);
    gx.set_xmx(l, P7G_B, NEG_INF);
    gx.set_xmx(l, P7G_N, NEG_INF);
    gx.set_xmx(l, P7G_C, gm.xsc[P7P_C][P7P_MOVE]);
    gx.set_xmx(l, P7G_E, gx.xmx(l, P7G_C) + gm.xsc[P7P_E][P7P_MOVE]);

    gx.set_mmx(l, mm, gx.xmx(l, P7G_E));
    gx.set_dmx(l, mm, gx.xmx(l, P7G_E));
    gx.set_imx(l, mm, NEG_INF);
    for k in (1..mm).rev() {
        let mval = p7_flogsum(gx.xmx(l, P7G_E) + esc, gx.dmx(l, k + 1) + tsc(gm, P7P_MD, k));
        gx.set_mmx(l, k, mval);
        let dval = p7_flogsum(gx.xmx(l, P7G_E) + esc, gx.dmx(l, k + 1) + tsc(gm, P7P_DD, k));
        gx.set_dmx(l, k, dval);
        gx.set_imx(l, k, NEG_INF);
    }

    // C 195-230: main recursion.
    for i in (1..l).rev() {
        let xip1 = dsq[i + 1] as usize;
        let rscx = &gm.rsc[xip1];

        // B state (C 199-201)
        let mut b = gx.mmx(i + 1, 1) + tsc(gm, P7P_BM, 0) + rscx[1 * P7P_NR + P7P_MSC];
        for k in 2..=mm {
            b = p7_flogsum(b, gx.mmx(i + 1, k) + tsc(gm, P7P_BM, k - 1) + rscx[k * P7P_NR + P7P_MSC]);
        }
        gx.set_xmx(i, P7G_B, b);

        // J state (C 203-204)
        let j = p7_flogsum(
            gx.xmx(i + 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
            gx.xmx(i, P7G_B) + gm.xsc[P7P_J][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_J, j);

        // C state (C 206)
        let c = gx.xmx(i + 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP];
        gx.set_xmx(i, P7G_C, c);

        // E state (C 208-209)
        let e = p7_flogsum(
            gx.xmx(i, P7G_J) + gm.xsc[P7P_E][P7P_LOOP],
            gx.xmx(i, P7G_C) + gm.xsc[P7P_E][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_E, e);

        // N state (C 211) — N<-N loop and N<-B move
        let n = p7_flogsum(
            gx.xmx(i + 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP],
            gx.xmx(i, P7G_B) + gm.xsc[P7P_N][P7P_MOVE],
        );
        gx.set_xmx(i, P7G_N, n);

        // M_M, D_M, I_M (C 215-216)
        gx.set_mmx(i, mm, gx.xmx(i, P7G_E));
        gx.set_dmx(i, mm, gx.xmx(i, P7G_E));
        gx.set_imx(i, mm, NEG_INF);

        for k in (1..mm).rev() {
            // M (C 219-222): FLogsum( FLogsum(MM+MSC, MI+ISC),
            //                         FLogsum(E+esc, MD) )  — keep exact nesting.
            let mval = p7_flogsum(
                p7_flogsum(
                    gx.mmx(i + 1, k + 1) + tsc(gm, P7P_MM, k) + rscx[(k + 1) * P7P_NR + P7P_MSC],
                    gx.imx(i + 1, k) + tsc(gm, P7P_MI, k) + rscx[k * P7P_NR + P7P_ISC],
                ),
                p7_flogsum(
                    gx.xmx(i, P7G_E) + esc,
                    gx.dmx(i, k + 1) + tsc(gm, P7P_MD, k),
                ),
            );
            gx.set_mmx(i, k, mval);

            // I (C 224-225)
            let ival = p7_flogsum(
                gx.mmx(i + 1, k + 1) + tsc(gm, P7P_IM, k) + rscx[(k + 1) * P7P_NR + P7P_MSC],
                gx.imx(i + 1, k) + tsc(gm, P7P_II, k) + rscx[k * P7P_NR + P7P_ISC],
            );
            gx.set_imx(i, k, ival);

            // D (C 227-229): FLogsum( DM+MSC, FLogsum(DD, E+esc) ) — keep nesting.
            let dval = p7_flogsum(
                gx.mmx(i + 1, k + 1) + tsc(gm, P7P_DM, k) + rscx[(k + 1) * P7P_NR + P7P_MSC],
                p7_flogsum(
                    gx.dmx(i, k + 1) + tsc(gm, P7P_DD, k),
                    gx.xmx(i, P7G_E) + esc,
                ),
            );
            gx.set_dmx(i, k, dval);
        }
    }

    // C 233-247: at i=0, only N,B states reachable.
    //   XMX(0,B) = MMX(1,1)+TSC(BM,0)+MSC(1); for k=2..M FLogsum in MMX(1,k)+TSC(BM,k-1)+MSC(k)
    //   XMX(0,J)=XMX(0,C)=XMX(0,E)=-inf;
    //   XMX(0,N) = FLogsum(XMX(1,N)+xsc[N][LOOP], XMX(0,B)+xsc[N][MOVE]);
    //   MMX(0,k)=IMX(0,k)=DMX(0,k)=-inf; *opt_sc = XMX(0,N)
    let x1 = dsq[1] as usize;
    let rsc1 = &gm.rsc[x1];
    let mut b0 = gx.mmx(1, 1) + tsc(gm, P7P_BM, 0) + rsc1[1 * P7P_NR + P7P_MSC];
    for k in 2..=mm {
        b0 = p7_flogsum(b0, gx.mmx(1, k) + tsc(gm, P7P_BM, k - 1) + rsc1[k * P7P_NR + P7P_MSC]);
    }
    gx.set_xmx(0, P7G_B, b0);
    gx.set_xmx(0, P7G_J, NEG_INF);
    gx.set_xmx(0, P7G_C, NEG_INF);
    gx.set_xmx(0, P7G_E, NEG_INF);
    let n0 = p7_flogsum(
        gx.xmx(1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP],
        gx.xmx(0, P7G_B) + gm.xsc[P7P_N][P7P_MOVE],
    );
    gx.set_xmx(0, P7G_N, n0);
    for k in 1..=mm {
        gx.set_mmx(0, k, NEG_INF);
        gx.set_imx(0, k, NEG_INF);
        gx.set_dmx(0, k, NEG_INF);
    }
    gx.xmx(0, P7G_N)
}

// ---------------------------------------------------------------------------
// c5: glocal domain definition (cm_p7_domaindef.c) — deterministic (single
// domain) path. RNG-based multidomain clustering is NOT ported here; callers
// must detect a multidomain region (is_multidomain_region) and handle it.
// ---------------------------------------------------------------------------

/// C `p7_ReconfigUnihit` (modelconfig.c): xsc[E][MOVE]=0, xsc[E][LOOP]=-inf,
/// nj=0, then ReconfigLength.
pub fn p7_reconfig_unihit(gm: &mut GlocalProfile, l: i32) {
    gm.xsc[P7P_E][P7P_MOVE] = 0.0;
    gm.xsc[P7P_E][P7P_LOOP] = NEG_INF;
    gm.nj = 0.0;
    reconfig_length(gm, l);
}

/// C `p7_ReconfigMultihit` (modelconfig.c): xsc[E][MOVE]=xsc[E][LOOP]=-log2,
/// nj=1, then ReconfigLength.
pub fn p7_reconfig_multihit(gm: &mut GlocalProfile, l: i32) {
    let log2 = std::f64::consts::LN_2 as f32;
    gm.xsc[P7P_E][P7P_MOVE] = -log2;
    gm.xsc[P7P_E][P7P_LOOP] = -log2;
    gm.nj = 1.0;
    reconfig_length(gm, l);
}

/// C `p7_GDomainDecoding` (generic_decoding.c:208): fills btot/etot/mocc arrays
/// (indices 0..=L) from the multihit Forward/Backward matrices. Returns
/// (btot, etot, mocc). NOTE: btot/etot accumulate `exp` (double), mocc uses
/// `expf` (float) — matched here for byte fidelity.
pub fn p7_gdomain_decoding(
    gm: &GlocalProfile,
    fwd: &P7Gmx,
    bck: &P7Gmx,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let l = fwd.l;
    let overall_logp = fwd.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE];
    let mut btot = vec![0.0f32; l + 1];
    let mut etot = vec![0.0f32; l + 1];
    let mut mocc = vec![0.0f32; l + 1];
    for i in 1..=l {
        // btot[i] = btot[i-1] + exp( fwd.B(i-1) + bck.B(i-1) - overall ) [double exp]
        btot[i] = btot[i - 1]
            + ((fwd.xmx(i - 1, P7G_B) + bck.xmx(i - 1, P7G_B) - overall_logp) as f64).exp() as f32;
        etot[i] = etot[i - 1]
            + ((fwd.xmx(i, P7G_E) + bck.xmx(i, P7G_E) - overall_logp) as f64).exp() as f32;
        // njcp uses expf (float)
        let mut njcp = (fwd.xmx(i - 1, P7G_N) + bck.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_LOOP] - overall_logp).exp();
        njcp += (fwd.xmx(i - 1, P7G_J) + bck.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_LOOP] - overall_logp).exp();
        njcp += (fwd.xmx(i - 1, P7G_C) + bck.xmx(i, P7G_C) + gm.xsc[P7P_C][P7P_LOOP] - overall_logp).exp();
        mocc[i] = 1.0 - njcp;
    }
    (btot, etot, mocc)
}

/// C `p7_GDecoding` (generic_decoding.c): overwrite `pp` with posterior
/// probabilities from Forward `fwd` and Backward `bck`.
pub fn p7_gdecoding(gm: &GlocalProfile, fwd: &P7Gmx, bck: &P7Gmx, pp: &mut P7Gmx) {
    let l = fwd.l;
    let mm = gm.m;
    let overall_sc = fwd.xmx(l, P7G_C) + gm.xsc[P7P_C][P7P_MOVE];
    pp.m = mm;
    pp.l = l;

    // row 0 = 0
    pp.set_xmx(0, P7G_E, 0.0);
    pp.set_xmx(0, P7G_N, 0.0);
    pp.set_xmx(0, P7G_J, 0.0);
    pp.set_xmx(0, P7G_B, 0.0);
    pp.set_xmx(0, P7G_C, 0.0);
    for k in 0..=mm {
        pp.set_mmx(0, k, 0.0);
        pp.set_imx(0, k, 0.0);
        pp.set_dmx(0, k, 0.0);
    }

    for i in 1..=l {
        let mut denom = 0.0f32;
        pp.set_mmx(i, 0, 0.0);
        pp.set_imx(i, 0, 0.0);
        pp.set_dmx(i, 0, 0.0);
        for k in 1..mm {
            let mv = (fwd.mmx(i, k) + bck.mmx(i, k) - overall_sc).exp();
            pp.set_mmx(i, k, mv);
            denom += mv;
            let iv = (fwd.imx(i, k) + bck.imx(i, k) - overall_sc).exp();
            pp.set_imx(i, k, iv);
            denom += iv;
            pp.set_dmx(i, k, 0.0);
        }
        let mvm = (fwd.mmx(i, mm) + bck.mmx(i, mm) - overall_sc).exp();
        pp.set_mmx(i, mm, mvm);
        denom += mvm;
        pp.set_imx(i, mm, 0.0);
        pp.set_dmx(i, mm, 0.0);

        pp.set_xmx(i, P7G_E, 0.0);
        let n = (fwd.xmx(i - 1, P7G_N) + bck.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_LOOP] - overall_sc).exp();
        pp.set_xmx(i, P7G_N, n);
        let j = (fwd.xmx(i - 1, P7G_J) + bck.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_LOOP] - overall_sc).exp();
        pp.set_xmx(i, P7G_J, j);
        pp.set_xmx(i, P7G_B, 0.0);
        let c = (fwd.xmx(i - 1, P7G_C) + bck.xmx(i, P7G_C) + gm.xsc[P7P_C][P7P_LOOP] - overall_sc).exp();
        pp.set_xmx(i, P7G_C, c);
        denom += n + j + c;

        let inv = 1.0 / denom;
        for k in 1..mm {
            pp.set_mmx(i, k, pp.mmx(i, k) * inv);
            pp.set_imx(i, k, pp.imx(i, k) * inv);
        }
        pp.set_mmx(i, mm, pp.mmx(i, mm) * inv);
        pp.set_xmx(i, P7G_N, pp.xmx(i, P7G_N) * inv);
        pp.set_xmx(i, P7G_J, pp.xmx(i, P7G_J) * inv);
        pp.set_xmx(i, P7G_C, pp.xmx(i, P7G_C) * inv);
    }
}

/// C `p7_GNull2_ByExpectation` (generic_null2.c): posterior-weighted null2
/// odds ratios. Uses `pp` row 0 as workspace (destroys it). Returns null2[0..Kp].
pub fn p7_gnull2_by_expectation(gm: &GlocalProfile, pp: &mut P7Gmx) -> [f32; KP] {
    let mm = gm.m;
    let ld = pp.l;
    let ncell = (mm + 1) * P7G_NSCELLS;

    // Sum expected counts into row 0 (dp[0]) and xmx[0].
    // esl_vec_FCopy(dp[1], ncell, dp[0]); FCopy(xmx+NX, NX, xmx);
    let (row0_start, row1_start) = (0usize, 1 * (mm + 1) * P7G_NSCELLS);
    for c in 0..ncell {
        pp.dp[row0_start + c] = pp.dp[row1_start + c];
    }
    for s in 0..P7G_NXCELLS {
        pp.xmx[s] = pp.xmx[P7G_NXCELLS + s];
    }
    // for i=2..Ld: FAdd(dp[0], dp[i]); FAdd(xmx, xmx+i*NX)
    for i in 2..=ld {
        let rowi = i * (mm + 1) * P7G_NSCELLS;
        for c in 0..ncell {
            pp.dp[row0_start + c] += pp.dp[rowi + c];
        }
        for s in 0..P7G_NXCELLS {
            pp.xmx[s] += pp.xmx[i * P7G_NXCELLS + s];
        }
    }
    // FLog + FIncrement(-log Ld)
    let neg_log_ld = -(ld as f32).ln();
    for c in 0..ncell {
        pp.dp[row0_start + c] = pp.dp[row0_start + c].ln() + neg_log_ld;
    }
    for s in 0..P7G_NXCELLS {
        pp.xmx[s] = pp.xmx[s].ln() + neg_log_ld;
    }

    // xfactor = FLogsum(FLogsum(N,C),J) from row 0
    let mut xfactor = pp.xmx(0, P7G_N);
    xfactor = p7_flogsum(xfactor, pp.xmx(0, P7G_C));
    xfactor = p7_flogsum(xfactor, pp.xmx(0, P7G_J));

    let mut null2 = [0.0f32; KP];
    for x in 0..K {
        null2[x] = NEG_INF;
    }
    for x in 0..K {
        for k in 1..mm {
            null2[x] = p7_flogsum(null2[x], pp.mmx(0, k) + gm.rsc[x][k * P7P_NR + P7P_MSC]);
            null2[x] = p7_flogsum(null2[x], pp.imx(0, k) + gm.rsc[x][k * P7P_NR + P7P_ISC]);
        }
        // C uses MSC(gm, k, x) with k==M after the loop
        null2[x] = p7_flogsum(null2[x], pp.mmx(0, mm) + gm.rsc[x][mm * P7P_NR + P7P_MSC]);
        null2[x] = p7_flogsum(null2[x], xfactor);
    }
    for x in 0..K {
        null2[x] = null2[x].exp();
    }
    // esl_abc_FAvgScVec: degenerate codes = unweighted mean of canonical odds.
    for x in (K + 1)..=(KP - 3) {
        let mut result = 0.0f32;
        let mut ndegen = 0.0f32;
        for i in 0..K {
            if RNA_DEGEN[x][i] {
                result += null2[i];
                ndegen += 1.0;
            }
        }
        null2[x] = if ndegen > 0.0 { result / ndegen } else { 0.0 };
    }
    null2[K] = 1.0; // gap
    null2[KP - 2] = 1.0; // nonresidue
    null2[KP - 1] = 1.0; // missing
    null2
}

/// A single domain/envelope produced by glocal domain definition.
#[derive(Clone, Debug)]
pub struct Domain {
    pub ienv: i64,          // envelope start (window-local, 1-based)
    pub jenv: i64,          // envelope end
    pub envsc: f32,         // envelope Forward score, NATS
    pub domcorrection: f32, // null2 correction, NATS
}

/// C `is_multidomain_region` (cm_p7_domaindef.c:270): max_z min(E(z),B(z)) >= rt3.
fn is_multidomain_region(btot: &[f32], etot: &[f32], i: usize, j: usize, rt3: f32) -> bool {
    let mut max = -1.0f32;
    for z in i..=j {
        let e = etot[z] - etot[i - 1];
        let b = btot[j] - btot[z - 1];
        let expected_n = if e < b { e } else { b };
        if expected_n > max {
            max = expected_n;
        }
    }
    max >= rt3
}

/// C `glocal_rescore_isolated_domain` (cm_p7_domaindef.c:498), single-domain
/// path with do_null2=TRUE, do_aln=FALSE. `dsq` is the whole window (sentinel
/// padded); i..j are window-local envelope coords; `gm` must be UNIHIT.
/// Returns the Domain (ienv/jenv/envsc/domcorrection).
fn glocal_rescore_isolated_domain(
    gm: &GlocalProfile,
    dsq: &[u8],
    i: usize,
    j: usize,
    do_null2: bool,
) -> Domain {
    let ld = j - i + 1;
    // C 511: p7_GForward(sq->dsq + i-1, Ld, gm, gx1, &envsc). Build a
    // sentinel-padded slice whose residue 1 == dsq[i].
    let mut sub = vec![255u8];
    sub.extend_from_slice(&dsq[i..=j]);
    sub.push(255u8);

    let mut gx1 = P7Gmx::new(gm.m, ld);
    let envsc = p7_gforward(&sub, ld, gm, &mut gx1);

    // C 513-533: only when (do_null2 || do_aln). do_aln is always FALSE here, so
    // this whole block (GBackward + GDecoding + GNull2 + domcorrection) runs
    // only when do_null2 is TRUE (i.e. cmsearch --null2). Default: domcorrection=0.
    let mut domcorrection = 0.0f32;
    if do_null2 {
        let mut gx2 = P7Gmx::new(gm.m, ld);
        p7_gbackward(&sub, ld, gm, &mut gx2);
        // C passes p7_GDecoding(gm, gx1, gx2, gx2): reads bck from gx2 then
        // overwrites gx2 in place. Row i of pp only depends on row i of fwd/bck,
        // so we snapshot bck first for a faithful (order-independent) transcription.
        let (dp, xmx) = gx2.clone_matrix();
        let bck = P7Gmx { m: gm.m, l: ld, dp, xmx };
        p7_gdecoding(gm, &gx1, &bck, &mut gx2);
        let null2 = p7_gnull2_by_expectation(gm, &mut gx2);
        // C 527-532: n2sc[pos] = logf(null2[dsq[pos]]); domcorrection = sum i..j.
        for pos in i..=j {
            domcorrection += null2[dsq[pos] as usize].ln();
        }
    }

    Domain {
        ienv: i as i64,
        jenv: j as i64,
        envsc,
        domcorrection,
    }
}

impl P7Gmx {
    fn clone_matrix(&self) -> (Vec<f32>, Vec<f32>) {
        (self.dp.clone(), self.xmx.clone())
    }
}

/// C `p7_domaindef_GlocalByPosteriorHeuristics` (cm_p7_domaindef.c:113),
/// deterministic (non-multidomain) path. `gm` enters MULTIHIT (as configured
/// for the F4 Forward), `gxf`/`gxb` are the multihit Forward/Backward matrices
/// over the whole window `dsq` (1..=l). Returns the list of domains.
///
/// Panics if a region is multidomain (RNG clustering not yet ported) — callers
/// on the current DB should not hit this for compact single-domain models; the
/// panic surfaces the case loudly instead of silently diverging.
// ===========================================================================
// Multidomain region resolution: stochastic-traceback ensemble + single-linkage
// clustering. Faithful port of cm_p7_domaindef.c glocal_region_trace_ensemble +
// hmmer p7_spensemble.c + generic_stotrace.c p7_GStochasticTrace + Easel Fast RNG
// (esl_randomness knuth). do_null2 default FALSE → null2 accumulation skipped.
// ===========================================================================

/// Easel "Fast" RNG (esl_randomness_CreateFast → knuth). Byte-exact.
struct EslRng {
    x: u32,
}
/// C: esl_mix3 (Bob Jenkins mix), easel.c:2433. u32 wrapping.
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
    /// C: esl_randomness_Init (esl_random.c:226, FAST branch).
    fn new(seed: u32) -> Self {
        let mut x = esl_mix3(seed, 87654321, 12345678);
        if x == 0 { x = 42; }
        EslRng { x }
    }
    /// C: knuth (esl_random.c:325): x = x*69069 + 1.
    fn knuth(&mut self) -> u32 {
        self.x = self.x.wrapping_mul(69069).wrapping_add(1);
        self.x
    }
    /// C: esl_random (esl_random.c:286): x / 2^32, in [0,1).
    fn random(&mut self) -> f64 {
        (self.knuth() as f64) / 4294967296.0
    }
    /// C: esl_rnd_FChoose (esl_random.c:850): one draw, walk cumulative.
    fn fchoose(&mut self, p: &[f32]) -> usize {
        let mut norm = 0.0f64;
        let mut sum = 0.0f64;
        let roll = self.random();
        for &v in p { norm += v as f64; }
        for (i, &v) in p.iter().enumerate() {
            sum += v as f64;
            if roll < sum / norm { return i; }
        }
        p.len() - 1 // unreached in practice
    }
}

/// C: esl_vec_FSum (esl_vectorops.c) — Kahan-compensated float sum.
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
/// C: esl_vec_FLogSum (esl_vectorops.c) — max + logf(Σ expf(v-max)), skip < max-50.
fn esl_vec_flogsum(v: &[f32]) -> f32 {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::INFINITY { return f32::INFINITY; }
    let mut sum = 0.0f32;
    for &x in v {
        if x > max - 50.0 { sum += (x - max).exp(); }
    }
    sum.ln() + max
}
/// C: esl_vec_FLogNorm (esl_vectorops.c): logsum → increment(-denom) → exp → norm.
fn esl_vec_flognorm(v: &mut [f32]) {
    let denom = esl_vec_flogsum(v);
    for x in v.iter_mut() { *x += -denom; }
    for x in v.iter_mut() { *x = x.exp(); }
    let sum = esl_vec_fsum(v);
    if sum != 0.0 {
        for x in v.iter_mut() { *x /= sum; }
    } else {
        let n = v.len() as f32;
        for x in v.iter_mut() { *x = 1.0 / n; }
    }
}

// trace state codes (private)
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

/// C: p7_GStochasticTrace (generic_stotrace.c:47), GLOCAL profile. Samples a
/// traceback from the Forward matrix <gx> over region length <lr>. Returns the
/// forward-ordered trace as (state, k, i) triples.
fn p7_gstochastic_trace(rng: &mut EslRng, lr: usize, gm: &GlocalProfile, gx: &P7Gmx) -> Vec<(u8, i32, i32)> {
    let m = gm.m as i32;
    let mut tr: Vec<(u8, i32, i32)> = Vec::new();
    let mut k: i32 = 0;
    let mut i: i32 = lr as i32;
    tr.push((ST_T, k, i));
    tr.push((ST_C, k, i));
    let mut sprv = ST_C;
    let ninf = f32::NEG_INFINITY;
    while sprv != ST_S {
        let last = tr[tr.len() - 1].0;
        let iu = i as usize;
        let im1 = (i - 1) as usize;
        let ku = k as usize;
        let km1 = (k - 1) as usize;
        let mu = m as usize;
        let scur: u8;
        match last {
            ST_C => {
                let mut sc = [
                    gx.xmx(im1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP],
                    gx.xmx(iu, P7G_E) + gm.xsc[P7P_E][P7P_MOVE],
                ];
                esl_vec_flognorm(&mut sc);
                scur = if rng.fchoose(&sc) == 0 { ST_C } else { ST_E };
            }
            ST_E => {
                // glocal: E comes from M_M or D_M only (k=M)
                k = m;
                let mut sc = [gx.mmx(iu, mu), gx.dmx(iu, mu)];
                esl_vec_flognorm(&mut sc);
                scur = if rng.fchoose(&sc) == 0 { ST_M } else { ST_D };
            }
            ST_M => {
                let mut sc = [
                    gx.xmx(im1, P7G_B) + tsc(gm, P7P_BM, km1),
                    gx.mmx(im1, km1) + tsc(gm, P7P_MM, km1),
                    gx.imx(im1, km1) + tsc(gm, P7P_IM, km1),
                    gx.dmx(im1, km1) + tsc(gm, P7P_DM, km1),
                ];
                esl_vec_flognorm(&mut sc);
                scur = match rng.fchoose(&sc) {
                    0 => ST_B,
                    1 => ST_M,
                    2 => ST_I,
                    _ => ST_D,
                };
                k -= 1;
                i -= 1;
            }
            ST_D => {
                let mut sc = [
                    gx.mmx(iu, km1) + tsc(gm, P7P_MD, km1),
                    gx.dmx(iu, km1) + tsc(gm, P7P_DD, km1),
                ];
                esl_vec_flognorm(&mut sc);
                scur = if rng.fchoose(&sc) == 0 { ST_M } else { ST_D };
                k -= 1;
            }
            ST_I => {
                let mut sc = [
                    gx.mmx(im1, ku) + tsc(gm, P7P_MI, ku),
                    gx.imx(im1, ku) + tsc(gm, P7P_II, ku),
                ];
                esl_vec_flognorm(&mut sc);
                scur = if rng.fchoose(&sc) == 0 { ST_M } else { ST_I };
                i -= 1;
            }
            ST_N => {
                scur = if i == 0 { ST_S } else { ST_N };
            }
            ST_B => {
                let mut sc = [
                    gx.xmx(iu, P7G_N) + gm.xsc[P7P_N][P7P_MOVE],
                    gx.xmx(iu, P7G_J) + gm.xsc[P7P_J][P7P_MOVE],
                ];
                esl_vec_flognorm(&mut sc);
                scur = if rng.fchoose(&sc) == 0 { ST_N } else { ST_J };
            }
            ST_J => {
                let mut sc = [
                    gx.xmx(im1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
                    gx.xmx(iu, P7G_E) + gm.xsc[P7P_E][P7P_LOOP],
                ];
                esl_vec_flognorm(&mut sc);
                scur = if rng.fchoose(&sc) == 0 { ST_J } else { ST_E };
            }
            _ => unreachable!("bogus state in stochastic traceback"),
        }
        let _ = ninf;
        tr.push((scur, k, i));
        // deferred i decrement for NCJ loops
        if (scur == ST_N || scur == ST_J || scur == ST_C) && scur == sprv {
            i -= 1;
        }
        sprv = scur;
    }
    tr.reverse();
    tr
}

/// C: p7_trace_Index (p7_trace.c:1172). Returns per-domain (sqfrom,sqto,hmmfrom,hmmto).
fn trace_index(tr: &[(u8, i32, i32)]) -> Vec<(i32, i32, i32, i32)> {
    let mut doms: Vec<(i32, i32, i32, i32)> = Vec::new(); // (sqfrom,sqto,hmmfrom,hmmto)
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

/// one sampled segment pair (p7_spcoord_s subset)
#[derive(Clone, Copy)]
struct Seg { idx: i32, i: i32, j: i32, k: i32, m: i32 }

/// C: link_spsamples (p7_spensemble.c:190). min_overlap=0.8, of_smaller=TRUE, max_diagdiff=4.
fn link_spsamples(h1: &Seg, h2: &Seg) -> bool {
    let min_overlap = 0.8f32;
    let max_diagdiff = 4i32;
    // seq overlap
    let nov = h1.j.min(h2.j) - h1.i.max(h2.i) + 1;
    let n = (h1.j - h1.i + 1).min(h2.j - h2.i + 1);
    if (nov as f32) / (n as f32) < min_overlap { return false; }
    // hmm overlap (NOTE: no +1 in C)
    let nov = h1.m.min(h2.m) - h1.k.max(h2.k);
    let n = (h1.m - h1.k + 1).min(h2.m - h2.k + 1);
    if (nov as f32) / (n as f32) < min_overlap { return false; }
    // nearby diagonal test
    let (d1, d2) = (h1.i - h1.k, h2.i - h2.k);
    if (d1 - d2).abs() <= max_diagdiff { return true; }
    let (d1, d2) = (h1.j - h1.m, h2.j - h2.m);
    if (d1 - d2).abs() <= max_diagdiff { return true; }
    false
}

/// C: esl_cluster_SingleLinkage (esl_cluster.c:135). Returns (assignment, nc).
fn single_linkage(segs: &[Seg]) -> (Vec<usize>, usize) {
    let n = segs.len();
    let mut a: Vec<usize> = (0..n).map(|v| n - v - 1).collect(); // a[v] = n-v-1
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
            let mut i: i64 = na as i64 - 1;
            while i >= 0 {
                if link_spsamples(&segs[v], &segs[a[i as usize]]) {
                    let w = a[i as usize];
                    a[i as usize] = a[na - 1];
                    na -= 1;
                    b[nb] = w; nb += 1;
                }
                i -= 1;
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

/// C: p7_spensemble_Cluster (p7_spensemble.c:280) — significant clusters +
/// consensus endpoints. Returns significant clusters as (i,j,prob), sorted by i.
fn spensemble_cluster(segs: &[Seg], nsamples: usize) -> Vec<(i32, i32, f32)> {
    let min_posterior = 0.25f32;
    let min_endpointp = 0.02f32;
    if segs.is_empty() { return Vec::new(); }
    let (assignment, nc) = single_linkage(segs);
    let mut sigc: Vec<(i32, i32, i32, i32, f32)> = Vec::new(); // (i,j,k,m,prob)
    for c in 0..nc {
        // posterior prob: distinct trace-idx in cluster / nsamples
        let mut ninc = 0i32;
        let mut idx_of_last = -1i32;
        for h in 0..segs.len() {
            if assignment[h] == c {
                if segs[h].idx != idx_of_last { ninc += 1; }
                idx_of_last = segs[h].idx;
            }
        }
        if (ninc as f32) / (nsamples as f32) < min_posterior { continue; }
        // extent
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
        // leftmost i with enough endpoints
        let mut epc = vec![0i32; (imax - imin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].i - imin) as usize] += 1; } }
        let mut best_i = imin;
        while best_i <= imax { if epc[(best_i - imin) as usize] >= epc_threshold { break; } best_i += 1; }
        if best_i > imax { best_i = imin + iargmax(&epc) as i32; }
        // leftmost k
        let mut epc = vec![0i32; (kmax - kmin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].k - kmin) as usize] += 1; } }
        let mut best_k = kmin;
        while best_k <= kmax { if epc[(best_k - kmin) as usize] >= epc_threshold { break; } best_k += 1; }
        if best_k > kmax { best_k = kmin + iargmax(&epc) as i32; }
        // rightmost j
        let mut epc = vec![0i32; (jmax - jmin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].j - jmin) as usize] += 1; } }
        let mut best_j = jmax;
        while best_j >= jmin { if epc[(best_j - jmin) as usize] >= epc_threshold { break; } best_j -= 1; }
        if best_j < jmin { best_j = jmin + iargmax(&epc) as i32; }
        // rightmost m
        let mut epc = vec![0i32; (mmax - mmin + 1) as usize];
        for h in 0..segs.len() { if assignment[h] == c { epc[(segs[h].m - mmin) as usize] += 1; } }
        let mut best_m = mmax;
        while best_m >= mmin { if epc[(best_m - mmin) as usize] >= epc_threshold { break; } best_m -= 1; }
        if best_m < mmin { best_m = mmin + iargmax(&epc) as i32; }

        if best_i > best_j || best_k > best_m { continue; }
        sigc.push((best_i, best_j, best_k, best_m, ninc as f32 / nsamples as f32));
    }
    // sort by i (cluster_orderer)
    sigc.sort_by(|a, b| a.0.cmp(&b.0));
    // dominance removal (glocal_region_trace_ensemble)
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

/// C: glocal_region_trace_ensemble (cm_p7_domaindef.c:347). Resolve region
/// [ri..j] (coords local to <dsq>) into domain envelopes via stochastic-trace
/// clustering. Returns (i2,j2) domain coords (local to <dsq>), sorted by i.
/// do_null2 is FALSE by default → null2 accumulation is skipped entirely.
fn glocal_region_trace_ensemble(
    gm: &mut GlocalProfile,
    dsq: &[u8],
    ri: usize,
    j: usize,
    save_l: i32,
    nsamples: usize,
) -> Vec<(usize, usize)> {
    let seed = 181u32; // cmsearch --seed default; do_reseeding=TRUE resets per region
    let lr = j - ri + 1;

    // region Forward in MULTIHIT mode
    p7_reconfig_multihit(gm, save_l);
    let mut region = vec![255u8];
    region.extend_from_slice(&dsq[ri..=j]);
    region.push(255u8);
    let mut fwd = P7Gmx::new(gm.m, lr);
    let _fsc = p7_gforward(&region, lr, gm, &mut fwd);

    // ensemble of sampled tracebacks (reseed to original seed: do_reseeding)
    let mut rng = EslRng::new(seed);
    let mut segs: Vec<Seg> = Vec::new();
    for t in 0..nsamples {
        let tr = p7_gstochastic_trace(&mut rng, lr, gm, &fwd);
        let doms = trace_index(&tr);
        for (sqfrom, sqto, hmmfrom, hmmto) in doms {
            segs.push(Seg {
                idx: t as i32,
                i: sqfrom + ri as i32 - 1,
                j: sqto + ri as i32 - 1,
                k: hmmfrom,
                m: hmmto,
            });
        }
    }
    p7_reconfig_unihit(gm, save_l);

    let clusters = spensemble_cluster(&segs, nsamples);
    clusters.into_iter().map(|(i2, j2, _p)| (i2 as usize, j2 as usize)).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn p7_domaindef_glocal(
    gm: &mut GlocalProfile,
    dsq: &[u8],
    l: usize,
    gxf: &P7Gmx,
    gxb: &P7Gmx,
    do_null2: bool,
    rt1: f32,
    rt2: f32,
    rt3: f32,
    ns: usize,
    // C `save_mode_is_unihit` (cm_p7_domaindef.c:128): TRUE if the profile is
    // ALREADY unihit on entry. C then NEVER modifies its config (length nor mode) —
    // lines 139 and 228 are both gated by `!save_mode_is_unihit`. This matters for
    // the truncated passes (Rgm/Lgm/Tgm are UNILOCAL/UNIGLOCAL = unihit): their
    // 5'/3'-truncation length model (e.g. the 3' Lgm's C-state xC=-inf,0) must be
    // preserved through the isolated-domain rescore, NOT overwritten by the standard
    // `reconfig_length` a unihit reconfig would apply.
    is_unihit: bool,
) -> Vec<Domain> {
    let save_l = gm.l; // save length config (== wlen)

    // C 132: posterior decoding on the multihit matrices.
    let (btot, etot, mocc) = p7_gdomain_decoding(gm, gxf, gxb);

    // C 139: process each domain in unihit mode — but ONLY if the profile was NOT
    // already unihit on entry (else leave its config untouched, per C).
    if !is_unihit {
        p7_reconfig_unihit(gm, save_l);
    }

    let mut domains = Vec::new();
    // C 140-224: region-finding loop.
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
            // region i..j to evaluate
            let ri = i as usize;
            if is_multidomain_region(&btot, &etot, ri, j, rt3) {
                // C: resolve region into >=1 envelopes via stochastic-traceback
                // ensemble clustering (glocal_region_trace_ensemble).
                let clusters = glocal_region_trace_ensemble(gm, dsq, ri, j, save_l, ns);
                for (i2, j2) in clusters {
                    let dom = glocal_rescore_isolated_domain(gm, dsq, i2, j2, do_null2);
                    domains.push(dom);
                }
            } else {
                // single domain region → envelope
                let dom = glocal_rescore_isolated_domain(gm, dsq, ri, j, do_null2);
                domains.push(dom);
            }
            i = -1;
            triggered = false;
        }
    }

    // C 228-230: restore multihit — again only if we changed it (i.e. entered multihit).
    if !is_unihit {
        p7_reconfig_multihit(gm, save_l);
    }
    domains
}
