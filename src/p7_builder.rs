//! Faithful port of HMMER3's `p7_Builder` for Infernal cmbuild's DEFAULT filter
//! HMM path (the `p7_ARCH_HAND` branch used by `build_and_calibrate_p7_filter`).
//!
//! Byte-parity goal: the produced `P7Profile`'s transition parameters
//! `trans[k][0..7]`, `eff_nseq`, and `max_length` are identical to what C
//! cmbuild builds for the filter HMM (before its emissions are overwritten by
//! the marginalized-CM stage, which does not touch transitions/eff_nseq/maxl).
//!
//! C sources ported here (Infernal 1.1.5 + bundled HMMER3):
//!   hmmer/src/p7_builder.c  : p7_Builder, relative_weights, build_model,
//!                             effective_seqnumber, parameterize,
//!                             p7_Builder_MaxLength
//!   hmmer/src/build.c       : p7_Handmodelmaker, matassign2hmm
//!   hmmer/src/p7_trace.c    : p7_trace_FauxFromMSA, p7_trace_Doctor,
//!                             p7_trace_Count
//!   hmmer/src/p7_prior.c    : p7_ParameterEstimation
//!   hmmer/src/eweight.c     : p7_EntropyWeight (+ esl_root_Bisection)
//!   hmmer/src/p7_hmm.c      : p7_hmm_Scale, p7_hmm_Renormalize helpers
//!   hmmer/src/modelstats.c  : p7_MeanMatchRelativeEntropy
//!   easel/esl_msaweight.c   : esl_msaweight_PB_adv (ignore_rf=FALSE path)
//!   easel/esl_msa.c         : esl_msa_MarkFragments_old
//!   src/cmbuild.c           : cm_p7_prior_CreateNucleic, init_cfg, and the
//!                             amsa-RF construction in build_and_calibrate_p7_filter
//!
//! Every non-trivial block carries the C source it transcribes as a comment.

use crate::cm::CM;
use crate::p7_hmm::{P7Profile, P7H_MAP, P7H_RF, P7H_CS, P7H_MMASK, P7H_CONS, P7H_CHKSUM};
use crate::prior::Mixdchlet;
use crate::easel::msa::EslMsa;

// ---- Digital RNA alphabet codes (K=4, Kp=18), matching cm_modelmaker.rs ----
const K: usize = 4;
const KP: usize = 18;
const GAP: u8 = 4; // esl gap symbol '-'
const NONRESIDUE: u8 = 16; // '*'
const MISSING: u8 = 17; // '~' == esl_abc_XGetMissing(RNA)

// ---- p7 transition indices (hmmer.h:130-137) ----
const P7H_MM: usize = 0;
const P7H_MI: usize = 1;
const P7H_MD: usize = 2;
const P7H_IM: usize = 3;
const P7H_II: usize = 4;
const P7H_DM: usize = 5;
const P7H_DD: usize = 6;
const P7H_NTRANSITIONS: usize = 7;

// ---- Constants from cmbuild init_cfg / infernal.h / hmmer.h / easel.h ----
const DEFAULT_ETARGET_HMMFILTER: f64 = 0.38; // src/infernal.h:85 (fp7_bld->re_target)
const P7_DEFAULT_WINDOW_BETA: f64 = 1e-7; // hmmer.h:1262 (fp7_bld->w_beta)
const ESL_CONST_LOG2R: f64 = 1.44269504088896341; // easel.h:305
const P7_BUILDER_ESIGMA: f64 = 45.0; // p7_builder.c:113 (default esigma)
const FRAGTHRESH: f64 = 0.5; // p7_builder.c:114 (default fragthresh)

// =============================================================================
// Alphabet residue classification (esl_alphabet.h macros).
// =============================================================================
#[inline]
fn xis_residue(x: u8) -> bool {
    // esl_abc_XIsResidue: (x < K) || (x > K && x < Kp-2)
    (x as usize) < K || ((x as usize) > K && (x as usize) < KP - 2)
}
#[inline]
fn xis_gap(x: u8) -> bool {
    x == GAP // esl_abc_XIsGap: x == K
}
#[inline]
fn xis_missing(x: u8) -> bool {
    x == MISSING // esl_abc_XIsMissing: x == Kp-1
}
#[inline]
fn xis_nonresidue(x: u8) -> bool {
    x == NONRESIDUE // esl_abc_XIsNonresidue: x == Kp-2 ('*')
}
#[inline]
fn xis_canonical(x: u8) -> bool {
    (x as usize) < K
}
#[inline]
fn cis_gap(c: u8) -> bool {
    // esl_abc_CIsGap for nucleic: '-', '_', '.'
    c == b'.' || c == b'-' || c == b'_'
}

// =============================================================================
// Small vector ops (faithful esl_vectorops.c). Float precision matters.
// =============================================================================

/// esl_vec_FScale(vec,n,scale): vec[i] *= scale  (scale is a float).
#[inline]
fn f_scale(vec: &mut [f32], n: usize, scale: f32) {
    for x in vec.iter_mut().take(n) {
        *x *= scale;
    }
}

/// esl_vec_FSum(vec,n): Kahan compensated summation in f32.
#[inline]
fn f_sum(vec: &[f32], n: usize) -> f32 {
    let mut sum: f32 = 0.0;
    let mut c: f32 = 0.0;
    for &v in vec.iter().take(n) {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// esl_vec_FNorm(vec,n): normalize to sum 1; if sum==0 set uniform 1/n.
#[inline]
fn f_norm(vec: &mut [f32], n: usize) {
    let sum = f_sum(vec, n);
    if sum != 0.0 {
        for x in vec.iter_mut().take(n) {
            *x /= sum;
        }
    } else {
        let val = (1.0f64 / (n as f32 as f64)) as f32;
        for x in vec.iter_mut().take(n) {
            *x = val;
        }
    }
}

/// esl_vec_FSet(vec,n,value).
#[inline]
fn f_set(vec: &mut [f32], n: usize, value: f32) {
    for x in vec.iter_mut().take(n) {
        *x = value;
    }
}

/// esl_vec_FRelEntropy(p,q,n): sum_i p[i]*log2(p[i]/q[i]) in bits.
/// p[i]/q[i] is an f32 division, promoted to f64 for log2, accumulated in f32.
#[inline]
fn f_rel_entropy(p: &[f32], q: &[f32], n: usize) -> f32 {
    let mut kl: f32 = 0.0;
    for i in 0..n {
        let pi = p[i];
        if pi > 0.0 {
            let qi = q[i];
            if qi == 0.0 {
                return f32::INFINITY;
            } else {
                let ratio: f32 = pi / qi;
                kl += pi * (ratio as f64).log2() as f32;
            }
        }
    }
    kl
}

/// esl_vec_DNorm(vec,n): normalize a double vector to sum 1 (esl_vec_DSum is Kahan).
#[inline]
fn d_kahan_sum(vec: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &v in vec {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}
#[inline]
fn d_norm(vec: &mut [f64]) {
    let sum = d_kahan_sum(vec);
    if sum != 0.0 {
        for x in vec.iter_mut() {
            *x /= sum;
        }
    } else {
        let val = 1.0 / vec.len() as f64;
        for x in vec.iter_mut() {
            *x = val;
        }
    }
}

/// esl_abc_FCount into a K-length counter (esl_alphabet.c). Canonical adds wt;
/// gap/missing/nonresidue add nothing to the K canonical bins; a degenerate
/// residue distributes wt/ndegen across the canonical residues it represents.
fn fcount(ct: &mut [f32], x: u8, wt: f32) {
    if xis_canonical(x) {
        ct[x as usize] += wt;
    } else if xis_gap(x) || xis_missing(x) || xis_nonresidue(x) {
        // nothing (esl_abc_FCount adds to the gap/missing bin, unused for K-vector)
    } else {
        let nd = crate::cm::RNA_NDEGEN[x as usize] as f32;
        for y in 0..K {
            if crate::cm::RNA_DEGEN[x as usize][y] {
                ct[y] += wt / nd;
            }
        }
    }
}

// =============================================================================
// cm_p7_prior_CreateNucleic  (src/cmbuild.c:3396)
// =============================================================================

/// The Infernal-specific p7 Dirichlet prior used for filter HMMs.
pub struct P7Prior {
    pub tm: Mixdchlet, // match transitions: 1 comp, 3 params
    pub ti: Mixdchlet, // insert transitions: 1 comp, 2 params
    pub td: Mixdchlet, // delete transitions: 1 comp, 2 params
    pub em: Mixdchlet, // match emissions: 4 comp, 4 params
    pub ei: Mixdchlet, // insert emissions: 1 comp, 4 params
}

/// Faithful port of cm_p7_prior_CreateNucleic() (src/cmbuild.c:3396). Values
/// transcribed verbatim.
pub fn cm_p7_prior_create_nucleic() -> P7Prior {
    let num_comp = 4usize;
    // static double defmq[5] = { 0.079226, 0.259549, 0.241578, 0.419647 };
    let defmq: [f64; 4] = [0.079226, 0.259549, 0.241578, 0.419647];
    // static double defm[4][4]
    let defm: [[f64; 4]; 4] = [
        [1.294511, 0.400028, 6.579555, 0.509916],
        [0.090031, 0.028634, 0.086396, 0.041186],
        [0.158085, 0.448297, 0.114815, 0.394151],
        [1.740028, 1.487773, 1.565443, 1.947555],
    ];

    let mut tm = Mixdchlet::create(1, 3);
    let mut ti = Mixdchlet::create(1, 2);
    let mut td = Mixdchlet::create(1, 2);
    let mut em = Mixdchlet::create(num_comp, 4);
    let mut ei = Mixdchlet::create(1, 4);

    // Transition priors (from hmmer p7_prior_CreateNucleic).
    tm.q[0] = 1.0;
    tm.alpha[0][0] = 2.0; // TMM
    tm.alpha[0][1] = 0.1; // TMI
    tm.alpha[0][2] = 0.1; // TMD

    ti.q[0] = 1.0;
    ti.alpha[0][0] = 0.06; // TIM
    ti.alpha[0][1] = 0.2; // TII

    td.q[0] = 1.0;
    td.alpha[0][0] = 0.1; // TDM
    td.alpha[0][1] = 0.2; // TDD

    // Match emission priors.
    for q in 0..num_comp {
        em.q[q] = defmq[q];
        for a in 0..4 {
            em.alpha[q][a] = defm[q][a];
        }
    }

    // Insert emission prior: flat +1.
    ei.q[0] = 1.0;
    for a in 0..4 {
        ei.alpha[0][a] = 1.0;
    }

    P7Prior { tm, ti, td, em, ei }
}

// =============================================================================
// Counts-form P7 HMM (mirrors the count phase of C's P7_HMM).
// =============================================================================
#[derive(Clone)]
struct P7HmmCounts {
    m: usize,
    mat: Vec<[f32; K]>, // [0..M]; mat[0] special
    ins: Vec<[f32; K]>, // [0..M]
    t: Vec<[f32; P7H_NTRANSITIONS]>, // [0..M]
    nseq: i32,
}

impl P7HmmCounts {
    fn zero(m: usize) -> Self {
        // p7_hmm_Create + p7_hmm_Zero (p7_hmm.c): all vectors 0.
        P7HmmCounts {
            m,
            mat: vec![[0.0; K]; m + 1],
            ins: vec![[0.0; K]; m + 1],
            t: vec![[0.0; P7H_NTRANSITIONS]; m + 1],
            nseq: 0,
        }
    }
}

// =============================================================================
// esl_msa_MarkFragments_old  (easel/esl_msa.c:2205)
// =============================================================================

/// esl_abc_dsqrlen: number of residue positions in ax (1..alen).
fn dsq_rlen(ax: &[u8], alen: usize) -> i64 {
    let mut n: i64 = 0;
    for pos in 1..=alen {
        if xis_residue(ax[pos]) {
            n += 1;
        }
    }
    n
}

/// Faithful port of esl_msa_MarkFragments_old(msa, fragthresh) (digital mode).
/// A seq is a fragment iff rlen <= fragthresh*alen. For fragments, leading and
/// trailing non-residue symbols are replaced with the missing-data symbol.
fn mark_fragments_old(msa: &mut EslMsa, fragthresh: f64) {
    let alen = msa.alen as usize;
    for i in 0..msa.nseq {
        let rlen = dsq_rlen(&msa.ax[i], alen) as f64;
        if rlen <= fragthresh * alen as f64 {
            // for (pos=1; pos<=alen; pos++) { if IsResidue break; ax=missing; }
            for pos in 1..=alen {
                if xis_residue(msa.ax[i][pos]) {
                    break;
                }
                msa.ax[i][pos] = MISSING;
            }
            for pos in (1..=alen).rev() {
                if xis_residue(msa.ax[i][pos]) {
                    break;
                }
                msa.ax[i][pos] = MISSING;
            }
        }
    }
}

// =============================================================================
// relative_weights: esl_msaweight_PB_adv with ignore_rf=FALSE
//   (easel/esl_msaweight.c:182; consensus_by_rf + collect_counts + PB rule)
// =============================================================================

/// PB weighting restricted to RF consensus columns (ignore_rf=FALSE path used by
/// p7_ARCH_HAND). Requires msa.rf to be present. Sets msa.wgt (sums to nseq).
fn msaweight_pb_rf(msa: &mut EslMsa) {
    let alen = msa.alen as usize;
    let nseq = msa.nseq;
    if nseq == 1 {
        msa.wgt[0] = 1.0;
        return;
    }

    // ct[apos=0..alen][a=0..Kp-1]
    let mut ct = vec![vec![0i32; KP]; alen + 1];

    // consensus_by_rf(): conscols = list of apos where rf[apos-1] is not gap.
    let rf = msa.rf.as_ref().expect("msaweight_pb_rf requires RF").as_bytes().to_vec();
    let mut conscols: Vec<usize> = Vec::new();
    for apos in 1..=alen {
        if cis_gap(rf[apos - 1]) {
            continue;
        }
        conscols.push(apos);
    }
    let mut ncons = conscols.len();

    // collect_counts(): fragment-aware, only over consensus columns (ncons>0).
    // minspan = ceil(fragthresh * alen), fragthresh = eslMSAWEIGHT_FRAGTHRESH=0.5
    let fragthresh = 0.5f32;
    let minspan = (fragthresh * alen as f32).ceil() as i32;
    for idx in 0..nseq {
        let mut lpos = 1i32;
        while lpos <= alen as i32 && !xis_residue(msa.ax[idx][lpos as usize]) {
            lpos += 1;
        }
        let mut rpos = alen as i32;
        while rpos >= 1 && !xis_residue(msa.ax[idx][rpos as usize]) {
            rpos -= 1;
        }
        if rpos - lpos + 1 >= minspan {
            lpos = 1;
            rpos = alen as i32;
        }
        // ncons>0 branch: for (j=0; j<ncons && conscols[j] <= rpos; j++) if apos>=lpos count
        for j in 0..ncons {
            let apos = conscols[j] as i32;
            if apos > rpos {
                break;
            }
            if apos < lpos {
                continue;
            }
            let a = msa.ax[idx][apos as usize] as usize;
            ct[apos as usize][a] += 1;
        }
    }

    // (ncons is already >0 here; the consensus_by_all fallback is unreachable.)
    let _ = &mut ncons;

    // r[j] = number of distinct canonical residues in consensus column j.
    let mut r = vec![0i32; ncons];
    for j in 0..ncons {
        let apos = conscols[j];
        for a in 0..K {
            if ct[apos][a] > 0 {
                r[j] += 1;
            }
        }
    }

    // PB weight rule.
    for idx in 0..nseq {
        msa.wgt[idx] = 0.0;
        let mut rlen = 0i32;
        for j in 0..ncons {
            let apos = conscols[j];
            let a = msa.ax[idx][apos] as usize;
            if a >= K {
                // no contribution
            } else {
                msa.wgt[idx] += 1.0 / ((r[j] * ct[apos][a]) as f64);
                rlen += 1;
            }
        }
        if rlen > 0 {
            msa.wgt[idx] /= rlen as f64;
        }
    }

    // Normalize to sum to N.
    d_norm(&mut msa.wgt);
    for w in msa.wgt.iter_mut() {
        *w *= nseq as f64;
    }
}

// =============================================================================
// p7 traces + counting (p7_trace.c)
// =============================================================================

// p7 trace state types (p7_trace.h): only the ones we use.
const P7T_B: u8 = 1;
const P7T_M: u8 = 2;
const P7T_I: u8 = 3;
const P7T_D: u8 = 4;
const P7T_E: u8 = 5;
const P7T_X: u8 = 9; // missing-data / fragment

struct Trace {
    st: Vec<u8>,
    k: Vec<i32>,
    i: Vec<i32>,
}
impl Trace {
    fn new() -> Self {
        Trace { st: Vec::new(), k: Vec::new(), i: Vec::new() }
    }
    #[inline]
    fn append(&mut self, st: u8, k: i32, i: i32) {
        self.st.push(st);
        self.k.push(k);
        self.i.push(i);
    }
    #[inline]
    fn n(&self) -> usize {
        self.st.len()
    }
}

/// p7_trace_FauxFromMSA(msa, matassign, p7_MSA_COORDS, tr) (p7_trace.c:1277).
/// matassign is 1..alen bit flags. Returns one Trace per sequence, using MSA
/// (apos) coordinates for the emitted residue positions.
fn faux_from_msa(msa: &EslMsa, matassign: &[i32]) -> Vec<Trace> {
    let alen = msa.alen as usize;
    let mut trs: Vec<Trace> = Vec::with_capacity(msa.nseq);
    for idx in 0..msa.nseq {
        let mut tr = Trace::new();
        tr.append(P7T_B, 0, 0);
        let ax = &msa.ax[idx];
        let mut k = 0i32;
        for apos in 1..=alen {
            let showpos = apos as i32; // p7_MSA_COORDS
            let x = ax[apos];
            if matassign[apos] != 0 {
                // match or delete
                k += 1;
                if xis_residue(x) {
                    tr.append(P7T_M, k, showpos);
                } else if xis_gap(x) {
                    tr.append(P7T_D, k, 0);
                } else if xis_nonresidue(x) {
                    tr.append(P7T_M, k, showpos); // treat '*' as residue
                } else if xis_missing(x) {
                    if tr.st[tr.n() - 1] != P7T_X {
                        tr.append(P7T_X, k, 0);
                    }
                }
            } else {
                // insert or nothing
                if xis_residue(x) {
                    tr.append(P7T_I, k, showpos);
                } else if xis_nonresidue(x) {
                    tr.append(P7T_I, k, showpos);
                } else if xis_missing(x) {
                    if tr.st[tr.n() - 1] != P7T_X {
                        tr.append(P7T_X, k, 0);
                    }
                }
                // else gap: nothing
            }
        }
        tr.append(P7T_E, 0, 0);
        trs.push(tr);
    }
    trs
}

/// p7_trace_Doctor(tr) (p7_trace.c:1366): collapse illegal D->I / I->D into M.
fn trace_doctor(tr: &mut Trace) {
    let mut opos = 0usize;
    let mut npos = 0usize;
    let n = tr.n();
    // Work on copies because we overwrite in place left-to-right (npos <= opos).
    let st = tr.st.clone();
    let k = tr.k.clone();
    let i = tr.i.clone();
    while opos < n {
        if st[opos] == P7T_D && opos + 1 < n && st[opos + 1] == P7T_I {
            tr.st[npos] = P7T_M;
            tr.k[npos] = k[opos]; // D->M
            tr.i[npos] = i[opos + 1]; // insert char moves back
            opos += 2;
            npos += 1;
        } else if st[opos] == P7T_I && opos + 1 < n && st[opos + 1] == P7T_D {
            tr.st[npos] = P7T_M;
            tr.k[npos] = k[opos + 1]; // D->M
            tr.i[npos] = i[opos]; // insert char moves up
            opos += 2;
            npos += 1;
        } else {
            tr.st[npos] = st[opos];
            tr.k[npos] = k[opos];
            tr.i[npos] = i[opos];
            opos += 1;
            npos += 1;
        }
    }
    tr.st.truncate(npos);
    tr.k.truncate(npos);
    tr.i.truncate(npos);
}

/// p7_trace_Count(hmm, dsq, wt, tr) (p7_trace.c:1454). Accumulates weighted
/// emission and transition counts into `h`. `dsq` = msa.ax[idx].
fn trace_count(h: &mut P7HmmCounts, dsq: &[u8], wt: f32, tr: &Trace) {
    let n = tr.n();
    // Fragment bounds z1..z2 (skip incomplete flanking insertions).
    let mut z1 = 0usize;
    let mut z2 = n - 1;
    if tr.st[0] == P7T_B && n > 1 && tr.st[1] == P7T_X {
        let mut z = 2;
        while z < n - 1 {
            if tr.st[z] == P7T_M {
                z1 = z;
                break;
            }
            z += 1;
        }
    }
    if tr.st[n - 1] == P7T_E && n >= 2 && tr.st[n - 2] == P7T_X {
        let mut z = n as isize - 3;
        while z > 0 {
            if tr.st[z as usize] == P7T_M {
                z2 = z as usize;
                break;
            }
            z -= 1;
        }
    }

    let mut z = z1;
    while z < z2 {
        if tr.st[z] == P7T_X {
            z += 1;
            continue;
        }
        let st = tr.st[z];
        let st2 = tr.st[z + 1];
        let k = tr.k[z] as usize;
        let i = tr.i[z] as usize;

        // Emission counts.
        if st == P7T_M {
            fcount(&mut h.mat[k], dsq[i], wt);
        } else if st == P7T_I {
            fcount(&mut h.ins[k], dsq[i], wt);
        }

        // Transition counts.
        if st2 == P7T_X {
            z += 1;
            continue; // ignore transition to missing data
        }

        if st == P7T_B {
            let k2 = tr.k[z + 1] as usize;
            if st2 == P7T_M && k2 > 1 {
                // wing-retracted B->DD->Mk path
                h.t[0][P7H_MD] += wt;
                let mut ktmp = 1usize;
                while ktmp < k2 - 1 {
                    h.t[ktmp][P7H_DD] += wt;
                    ktmp += 1;
                }
                h.t[ktmp][P7H_DM] += wt;
            } else {
                match st2 {
                    P7T_M => h.t[0][P7H_MM] += wt,
                    P7T_I => h.t[0][P7H_MI] += wt,
                    P7T_D => h.t[0][P7H_MD] += wt,
                    _ => panic!("bad transition in trace"),
                }
            }
        } else if st == P7T_M {
            match st2 {
                P7T_M => h.t[k][P7H_MM] += wt,
                P7T_I => h.t[k][P7H_MI] += wt,
                P7T_D => h.t[k][P7H_MD] += wt,
                P7T_E => h.t[k][P7H_MM] += wt, // k==M
                _ => panic!("bad transition in trace"),
            }
        } else if st == P7T_I {
            match st2 {
                P7T_M => h.t[k][P7H_IM] += wt,
                P7T_I => h.t[k][P7H_II] += wt,
                P7T_E => h.t[k][P7H_IM] += wt, // k==M
                _ => panic!("bad transition in trace"),
            }
        } else if st == P7T_D {
            match st2 {
                P7T_M => h.t[k][P7H_DM] += wt,
                P7T_D => h.t[k][P7H_DD] += wt,
                P7T_E => h.t[k][P7H_DM] += wt, // k==M
                _ => panic!("bad transition in trace"),
            }
        }
        z += 1;
    }
}

// =============================================================================
// build_model -> p7_Handmodelmaker (build.c:80) + matassign2hmm (build.c:257)
// =============================================================================

/// Returns the counts-form HMM built from the msa's RF annotation.
fn build_model(msa: &EslMsa) -> P7HmmCounts {
    let alen = msa.alen as usize;
    let rf = msa.rf.as_ref().expect("build_model requires RF").as_bytes().to_vec();

    // matassign[apos] = CIsGap(rf[apos-1]) ? FALSE : TRUE   (build.c:93-94)
    let mut matassign = vec![0i32; alen + 1];
    for apos in 1..=alen {
        matassign[apos] = if cis_gap(rf[apos - 1]) { 0 } else { 1 };
    }

    // do_modelmask: msa->mm is None for these MSAs (skip).

    // How many match states? (build.c:272)
    let mut m = 0usize;
    for apos in 1..=alen {
        if matassign[apos] != 0 {
            m += 1;
        }
    }

    // Fake tracebacks + doctor + count (build.c:277-292).
    let mut trs = faux_from_msa(msa, &matassign);
    for tr in trs.iter_mut() {
        trace_doctor(tr);
    }

    let mut h = P7HmmCounts::zero(m);
    for idx in 0..msa.nseq {
        // C: p7_trace_Count(hmm, msa->ax[idx], msa->wgt[idx], tr[idx]);
        //    msa->wgt[idx] is double, truncated to float wt.
        let wt = msa.wgt[idx] as f32;
        trace_count(&mut h, &msa.ax[idx], wt, &trs[idx]);
    }
    h.nseq = msa.nseq as i32;
    h
}

// =============================================================================
// p7_hmm_Scale (p7_hmm.c:757) and clone-of-counts (p7_hmm_CopyParameters)
// =============================================================================

/// p7_hmm_Scale(hmm, scale): scale core counts. `scale` is passed to
/// esl_vec_FScale as a float (C signature), so the double is truncated first.
fn hmm_scale(h: &mut P7HmmCounts, scale: f64) {
    let s = scale as f32;
    for k in 0..=h.m {
        f_scale(&mut h.t[k], P7H_NTRANSITIONS, s);
        f_scale(&mut h.mat[k], K, s);
        f_scale(&mut h.ins[k], K, s);
    }
}

// =============================================================================
// parameterize -> p7_ParameterEstimation (p7_prior.c:297)
// =============================================================================

fn parameter_estimation(h: &mut P7HmmCounts, pri: &P7Prior) {
    let m = h.m;
    let mut c = [0.0f64; K];
    let mut p = [0.0f64; K];

    // Match transitions 0..M (first 3 of t[k]); TMD at node M is 0.
    for k in 0..=m {
        for a in 0..3 {
            c[a] = h.t[k][a] as f64;
        }
        pri.tm.mp_parameters(&c[..3], &mut p[..3]);
        for a in 0..3 {
            h.t[k][a] = p[a] as f32;
        }
    }
    h.t[m][P7H_MD] = 0.0;
    f_norm(&mut h.t[m][..3], 3);

    // Insert transitions 0..M (t[k]+3, i.e. IM,II).
    for k in 0..=m {
        c[0] = h.t[k][3] as f64;
        c[1] = h.t[k][4] as f64;
        pri.ti.mp_parameters(&c[..2], &mut p[..2]);
        h.t[k][3] = p[0] as f32;
        h.t[k][4] = p[1] as f32;
    }

    // Delete transitions 1..M-1 (t[k]+5, i.e. DM,DD).
    for k in 1..m {
        c[0] = h.t[k][5] as f64;
        c[1] = h.t[k][6] as f64;
        pri.td.mp_parameters(&c[..2], &mut p[..2]);
        h.t[k][5] = p[0] as f32;
        h.t[k][6] = p[1] as f32;
    }
    h.t[0][P7H_DM] = 1.0;
    h.t[m][P7H_DM] = 1.0;
    h.t[0][P7H_DD] = 0.0;
    h.t[m][P7H_DD] = 0.0;

    // Match emissions 1..M.
    for k in 1..=m {
        for a in 0..K {
            c[a] = h.mat[k][a] as f64;
        }
        pri.em.mp_parameters(&c[..K], &mut p[..K]);
        for a in 0..K {
            h.mat[k][a] = p[a] as f32;
        }
    }
    f_set(&mut h.mat[0], K, 0.0);
    h.mat[0][0] = 1.0;

    // Insert emissions 0..M.
    for k in 0..=m {
        for a in 0..K {
            c[a] = h.ins[k][a] as f64;
        }
        pri.ei.mp_parameters(&c[..K], &mut p[..K]);
        for a in 0..K {
            h.ins[k][a] = p[a] as f32;
        }
    }
}

// =============================================================================
// p7_MeanMatchRelativeEntropy (modelstats.c:80)
// =============================================================================

fn mean_match_relative_entropy(h: &P7HmmCounts, bg: &[f32; K]) -> f64 {
    let mut kl: f64 = 0.0;
    for k in 1..=h.m {
        kl += f_rel_entropy(&h.mat[k], bg, K) as f64;
    }
    kl /= h.m as f64;
    kl
}

// =============================================================================
// p7_EntropyWeight (eweight.c:60) + esl_root_Bisection (esl_rootfinder.c)
// =============================================================================

/// eweight_target_f: rel_entropy(param(scale h_counts to Neff)) - etarget.
fn eweight_target_f(orig: &P7HmmCounts, pri: &P7Prior, bg: &[f32; K], etarget: f64, neff: f64) -> f64 {
    // p7_hmm_CopyParameters(hmm, h2): reset h2 from the original counts.
    let mut h2 = orig.clone();
    // p7_hmm_Scale(h2, Neff/nseq).
    hmm_scale(&mut h2, neff / orig.nseq as f64);
    parameter_estimation(&mut h2, pri);
    mean_match_relative_entropy(&h2, bg) - etarget
}

/// esl_root_Bisection with rootfinder defaults (rel_tol 1e-12, max 100 iters),
/// abs tol per caller. Mirrors eweight.rs's port of esl_rootfinder.c.
fn esl_root_bisection<F: FnMut(f64) -> f64>(mut f: F, xl_in: f64, xr_in: f64, abs_tol: f64) -> f64 {
    let rel_tolerance: f64 = 1e-12;
    let residual_tol: f64 = 0.;
    let max_iter: i32 = 100;

    let mut xl = xl_in;
    let mut xr = xr_in;
    let mut fl = f(xl);
    let mut _fr = f(xr);

    let mut iter: i32 = 0;
    let mut x: f64 = 0.;
    loop {
        iter += 1;
        if iter > max_iter {
            break;
        }
        x = (xl + xr) / 2.;
        let fx = f(x);
        let xmag = if xl < 0. && xr > 0. { 0. } else { x };
        if fx == 0. {
            break;
        }
        if ((xr - xl) < abs_tol + rel_tolerance * xmag) || fx.abs() < residual_tol {
            break;
        }
        if fl > 0. {
            if fx > 0. {
                xl = x;
                fl = fx;
            } else {
                xr = x;
                _fr = fx;
            }
        } else if fx < 0. {
            xl = x;
            fl = fx;
        } else {
            xr = x;
            _fr = fx;
        }
    }
    x
}

/// p7_EntropyWeight(hmm, bg, pri, etarget, &Neff): returns effective seq number.
fn entropy_weight(orig: &P7HmmCounts, bg: &[f32; K], pri: &P7Prior, etarget: f64) -> f64 {
    let mut neff = orig.nseq as f64;
    let fx = eweight_target_f(orig, pri, bg, etarget, neff);
    if fx > 0. {
        // esl_root_Bisection(R, 0., nseq, &Neff); abs tol 0.01.
        neff = esl_root_bisection(
            |x| eweight_target_f(orig, pri, bg, etarget, x),
            0.,
            orig.nseq as f64,
            0.01,
        );
    }
    neff
}

// =============================================================================
// p7_Builder_MaxLength (p7_builder.c:651)
// =============================================================================

/// Faithful port of p7_Builder_MaxLength(hmm, emit_thresh). Returns max_length.
/// Operates on the probability-form transitions `t` (index 1..M valid).
fn builder_max_length(t: &[[f32; P7H_NTRANSITIONS]], model_len: usize, emit_thresh: f64) -> i32 {
    if model_len == 1 {
        return 1;
    }
    let length_bound = (model_len).max((20 * model_len).min(100000)) as i32;
    let mut max_length = length_bound; // default if target never reached

    // I,M,D : [model_len+1][2]
    let n = model_len + 1;
    let mut ii = vec![[0.0f64; 2]; n];
    let mut mm = vec![[0.0f64; 2]; n];
    let mut dd = vec![[0.0f64; 2]; n];

    let tt = |k: usize, idx: usize| -> f64 { t[k][idx] as f64 };

    // 1st column (col=1).
    mm[1][0] = 1.0;
    ii[1][0] = 0.0;
    dd[1][0] = 0.0;
    mm[2][0] = 0.0;
    ii[2][0] = 0.0;
    dd[2][0] = tt(1, P7H_MD);
    for k in 3..=model_len {
        mm[k][0] = 0.0;
        ii[k][0] = 0.0;
        dd[k][0] = tt(k - 1, P7H_DD) * dd[k - 1][0];
    }

    // 2nd column (col=2).
    mm[1][1] = 0.0;
    dd[1][1] = 0.0;
    dd[2][1] = 0.0;
    ii[2][1] = 0.0;
    ii[1][1] = tt(1, P7H_MI) * mm[1][0];
    mm[2][1] = tt(1, P7H_MM) * mm[1][0];
    for k in 3..=model_len {
        mm[k][1] = tt(k - 1, P7H_DM) * dd[k - 1][0];
        ii[k][1] = 0.0;
        dd[k][1] = tt(k - 1, P7H_MD) * mm[k - 1][1] + tt(k - 1, P7H_DD) * dd[k - 1][1];
    }

    let mut p_sum = mm[model_len][0] + mm[model_len][1] + dd[model_len][0] + dd[model_len][1];

    // General case for remaining columns.
    let mut col_ptr = 0usize;
    let mut col = 3i32;
    while col <= length_bound {
        let prev_col_ptr = 1 - col_ptr;
        let mut surv = 0.0f64;
        mm[1][col_ptr] = 0.0;
        dd[1][col_ptr] = 0.0;
        ii[1][col_ptr] = tt(1, P7H_II) * ii[1][prev_col_ptr];
        surv += ii[1][col_ptr];

        for k in 2..=model_len {
            mm[k][col_ptr] = tt(k - 1, P7H_MM) * mm[k - 1][prev_col_ptr]
                + tt(k - 1, P7H_DM) * dd[k - 1][prev_col_ptr]
                + tt(k - 1, P7H_IM) * ii[k - 1][prev_col_ptr];
            ii[k][col_ptr] = tt(k, P7H_MI) * mm[k][prev_col_ptr] + tt(k, P7H_II) * ii[k][prev_col_ptr];
            dd[k][col_ptr] = tt(k - 1, P7H_MD) * mm[k - 1][col_ptr] + tt(k - 1, P7H_DD) * dd[k - 1][col_ptr];

            // if (k<=model_len) — always true here.
            surv += ii[k][col_ptr]
                + mm[k][col_ptr] * (1.0 - tt(k, P7H_MD))
                + dd[k][col_ptr] * (1.0 - tt(k, P7H_DD));
        }
        surv += mm[model_len][col_ptr] * tt(model_len, P7H_MD)
            + dd[model_len][col_ptr] * tt(model_len, P7H_DD)
            - ii[model_len][col_ptr];

        p_sum += mm[model_len][col_ptr] + dd[model_len][col_ptr];
        surv /= surv + p_sum;

        if surv < emit_thresh {
            max_length = col;
            break;
        }
        col_ptr = 1 - col_ptr;
        col += 1;
    }

    max_length
}

// =============================================================================
// esl_msa_Checksum (easel/esl_msa.c) — for the CKSUM header field.
// =============================================================================
fn msa_checksum(msa: &EslMsa) -> u32 {
    let mut val: u32 = 0;
    let alen = msa.alen as usize;
    for i in 0..msa.nseq {
        for pos in 1..=alen {
            val = val.wrapping_add(msa.ax[i][pos] as u32);
            val = val.wrapping_add(val << 10);
            val ^= val >> 6;
        }
    }
    val = val.wrapping_add(val << 3);
    val ^= val >> 11;
    val = val.wrapping_add(val << 15);
    val
}

// =============================================================================
// Public entry point: build the filter p7 HMM (p7_Builder driver).
// =============================================================================

/// Build the DEFAULT cmbuild filter HMM from a covariance model `cm` (only its
/// `map`/`clen`/`name` are used to define the amsa RF consensus columns) and the
/// MSA `msa` as passed to C's `build_and_calibrate_p7_filter` — i.e. AFTER the CM
/// build's relative-weighting and (span-based) fragment marking. `msa` is not
/// modified; an internal clone (`amsa`) is used.
///
/// Faithful to build_and_calibrate_p7_filter (cmbuild.c:2366-2390) followed by
/// p7_Builder (p7_builder.c:415). Emissions (mat/ins/compo) are computed
/// internally (needed for entropy weighting) but the caller is expected to
/// overwrite them; `trans`, `eff_nseq`, `max_length` are the byte-parity targets.
/// `p7ere` is the `--p7ere <x>` override for the filter HMM's minimum relative
/// entropy per position (C: fp7_bld->re_target, cmbuild.c:897); `None` uses the
/// DEFAULT_ETARGET_HMMFILTER (0.38).
pub fn build_filter_p7(cm: &CM, msa: &EslMsa, p7ere: Option<f64>) -> P7Profile {
    // C: fp7_bld->re_target = esl_opt_IsOn("--p7ere") ? <x> : DEFAULT_ETARGET_HMMFILTER.
    let re_target = p7ere.unwrap_or(DEFAULT_ETARGET_HMMFILTER);
    // --- Construct amsa (esl_msa_Clone + RF from cm->map). ---
    let mut amsa = msa.clone();
    // clear GA/TC/NC cutoffs (they pertain to the CM).
    for s in amsa.cutset.iter_mut() {
        *s = false;
    }
    // amsa->rf: init all '.', then 'x' at cm->map[cpos]-1 for cpos=1..clen.
    let alen = amsa.alen as usize;
    let mut rf = vec![b'.'; alen];
    let clen = cm.clen as usize;
    for cpos in 1..=clen {
        let apos0 = (cm.map[cpos] - 1) as usize; // off-by-one (0-based)
        rf[apos0] = b'x';
    }
    amsa.rf = Some(String::from_utf8(rf).unwrap());

    // --- p7_Builder pipeline (p7_builder.c:415) ---
    let bg: [f32; K] = [0.25, 0.25, 0.25, 0.25]; // p7_bg_Create(RNA): 1/K each.
    let prior = cm_p7_prior_create_nucleic();

    // esl_msa_Checksum (before any weighting mutation of the clone; the clone's
    // ax already carries the CM build's fragment marks — same as C).
    let checksum = msa_checksum(&amsa);

    // relative_weights: PB with ignore_rf=FALSE (arch=HAND). Sets amsa.wgt.
    msaweight_pb_rf(&mut amsa);

    // esl_msa_MarkFragments_old(amsa, fragthresh=0.5). Modifies amsa.ax.
    mark_fragments_old(&mut amsa, FRAGTHRESH);

    // build_model -> p7_Handmodelmaker: counts-form HMM.
    let mut h = build_model(&amsa);
    let m = h.m;
    let nseq = h.nseq;

    // effective_seqnumber (ENTROPY branch).
    // etarget = max(re_target, (esigma - log2R*log(2/(M*(M+1))))/M)
    let mf = m as f64;
    let mut etarget = (P7_BUILDER_ESIGMA
        - ESL_CONST_LOG2R * (2.0 / (mf * (mf + 1.0))).ln())
        / mf;
    if re_target > etarget {
        etarget = re_target;
    }
    let eff_nseq_d = entropy_weight(&h, &bg, &prior, etarget);
    // hmm->eff_nseq = eff_nseq (stored as float).
    let eff_nseq_f = eff_nseq_d as f32;
    // p7_hmm_Scale(hmm, hmm->eff_nseq / (double) hmm->nseq)  — reads the FLOAT eff_nseq.
    hmm_scale(&mut h, eff_nseq_f as f64 / nseq as f64);

    // parameterize -> p7_ParameterEstimation.
    parameter_estimation(&mut h, &prior);

    // (annotate: force masked positions to bg — msa has no MM here, skip.)

    // DNA/RNA max_length block: w_len<0, w_beta!=0 -> p7_Builder_MaxLength.
    let max_length = builder_max_length(&h.t, m, P7_DEFAULT_WINDOW_BETA);

    // --- Assemble the P7Profile. ---
    let mut hmm = P7Profile::new(m as i32);
    hmm.name = cm.name.clone();
    hmm.alph = "RNA".to_string();
    for k in 0..=m {
        for a in 0..K {
            hmm.mat[k][a] = h.mat[k][a];
            hmm.ins[k][a] = h.ins[k][a];
        }
        for ttx in 0..P7H_NTRANSITIONS {
            hmm.trans[k][ttx] = h.t[k][ttx];
        }
    }
    hmm.nseq = nseq;
    hmm.eff_nseq = eff_nseq_f;
    hmm.max_length = max_length;
    hmm.checksum = checksum;
    hmm.flags |= P7H_CHKSUM;

    hmm
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify the cm_p7_prior_CreateNucleic values are transcribed verbatim.
    #[test]
    fn prior_values() {
        let p = cm_p7_prior_create_nucleic();
        assert_eq!(p.tm.alpha[0], vec![2.0, 0.1, 0.1]);
        assert_eq!(p.ti.alpha[0], vec![0.06, 0.2]);
        assert_eq!(p.td.alpha[0], vec![0.1, 0.2]);
        assert_eq!(p.em.q, vec![0.079226, 0.259549, 0.241578, 0.419647]);
        assert_eq!(p.em.alpha[0], vec![1.294511, 0.400028, 6.579555, 0.509916]);
        assert_eq!(p.ei.alpha[0], vec![1.0, 1.0, 1.0, 1.0]);
    }

    // etarget dominance: esigma term for small M, re_target (0.38) for large M.
    #[test]
    fn etarget_regimes() {
        let et = |m: f64| {
            let e =
                (P7_BUILDER_ESIGMA - ESL_CONST_LOG2R * (2.0 / (m * (m + 1.0))).ln()) / m;
            if DEFAULT_ETARGET_HMMFILTER > e { DEFAULT_ETARGET_HMMFILTER } else { e }
        };
        assert!((et(72.0) - 0.78288).abs() < 1e-3); // tRNA5: esigma term wins
        assert!((et(379.0) - 0.38).abs() < 1e-9); // RNaseP: re_target wins
    }
}

/// Convenience: also return the final mean-match relative entropy (bits) of the
/// parameterized filter HMM (C debug `fhmm_re`), for verification.
pub fn build_filter_p7_with_re(cm: &CM, msa: &EslMsa, p7ere: Option<f64>) -> (P7Profile, f64) {
    let hmm = build_filter_p7(cm, msa, p7ere);
    // Recompute rel-entropy from the final probabilities against uniform bg.
    let bg: [f32; K] = [0.25, 0.25, 0.25, 0.25];
    let mut kl = 0.0f64;
    for k in 1..=hmm.m as usize {
        kl += f_rel_entropy(&hmm.mat[k], &bg, K) as f64;
    }
    kl /= hmm.m as f64;
    (hmm, kl)
}
