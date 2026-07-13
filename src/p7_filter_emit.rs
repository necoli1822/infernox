// SPDX-License-Identifier: BSD-3-Clause
//! Stage B: filter-HMM EMISSION construction for cmbuild's p7 filter.
//!
//! Faithful byte-parity port of the `!--p7hemit` (default) branch of Infernal
//! 1.1.5's `build_and_calibrate_p7_filter` (src/cmbuild.c:2402-2450), together
//! with the emission portion of `cm_cp9_to_p7` (src/cm_p7_modelmaker.c:118-225),
//! `p7_hmm_SetComposition` (hmmer/src/p7_hmm.c) and `cm_p7_hmm_SetConsensus`
//! (src/cm_p7_modelmaker.c:557).
//!
//! In C the filter HMM's match/insert emissions are NOT taken from the p7
//! builder; instead a *temporary* CM (`acm`) is built from the SAME MSA, but
//! entropy-weighted so that its mean match relative entropy matches the p7
//! filter HMM's (`fhmm_re`, computed in stage A). That temp CM is configured,
//! which builds its CP9 HMM (`build_cp9_hmm`) and, via `cm_cp9_to_p7`, its
//! maximum-likelihood p7 HMM (`acm->mlp7`). The mlp7 match/insert emissions are
//! then copied (and renormalized) onto the filter HMM `fhmm`.
//!
//! This module reproduces exactly that temp-CM construction (reusing the same
//! faithful modelmaker/prior/eweight/cp9 code that the cmbuild bin uses for the
//! CM body) and reads off the emission vectors + COMPO + CONS.
//!
//! ## C source, cmbuild.c:2407-2447 (the branch we port):
//! ```c
//!   build_model(go, cfg, errbuf, FALSE, msa, &acm, NULL, NULL);
//!   fhmm_re = p7_MeanMatchRelativeEntropy(fhmm, cfg->fp7_bg);
//!   cm_EntropyWeight(acm, cfg->pri, fhmm_re, esl_opt_GetReal(go,"--eminseq"),
//!                    (emaxseq used ? emaxseq : (double) cm->nseq),
//!                    TRUE, &mlp7_re, &neff);
//!   acm->eff_nseq = neff;
//!   cm_Rescale(acm, acm->eff_nseq / (float) msa->nseq);
//!   parameterize(go, cfg, errbuf, FALSE, acm, cfg->pri, msa->nseq);
//!   configure_model(go, cfg, errbuf, acm, 2);           // builds acm->cp9 + acm->mlp7
//!   for (k=1;k<=fhmm->M;k++) esl_vec_FCopy(acm->mlp7->mat[k], K, fhmm->mat[k]);
//!   for (k=1;k<=fhmm->M;k++) esl_vec_FNorm(fhmm->mat[k], K);
//!   esl_vec_FSet(fhmm->mat[0], K, 0.); fhmm->mat[0][0] = 1.0;
//!   for (k=0;k<=fhmm->M;k++) esl_vec_FCopy(acm->mlp7->ins[k], K, fhmm->ins[k]);
//!   for (k=0;k<=fhmm->M;k++) esl_vec_FNorm(fhmm->ins[k], K);
//!   p7_hmm_SetComposition(fhmm);
//!   fhmm->eff_nseq = acm->eff_nseq;
//! ```

use crate::cm::{ALPHABET_SIZE, CM};
use crate::cm_modelmaker as mm;
use crate::cp9;
use crate::p7_hmm::P7Profile;
use crate::easel::alphabet::EslAlphabet;
use crate::easel::msa::EslMsa;

/// Result of the stage-B filter-HMM emission construction.
pub struct FilterEmissions {
    /// Filter HMM match emissions `mat[0..=M]`, `mat[k][A,C,G,U]`. `mat[0] =
    /// {1,0,0,0}` (the B-state convention). Byte-identical to what C copies onto
    /// `fhmm->mat`.
    pub mat: Vec<[f32; ALPHABET_SIZE]>,
    /// Filter HMM insert emissions `ins[0..=M]` (all uniform 0.25 for a normal
    /// flattened build). Byte-identical to C's `fhmm->ins`.
    pub ins: Vec<[f32; ALPHABET_SIZE]>,
    /// Consensus residues `consensus[0..=M]` from `cm_p7_hmm_SetConsensus`
    /// (index 0 = ' '); uppercase where `mat[k][argmax] >= 0.5`, else lowercase.
    pub consensus: Vec<u8>,
    /// Temp-CM effective sequence number (`acm->eff_nseq`); C `neff`. Copied to
    /// `fhmm->eff_nseq`.
    pub neff: f64,
    /// Achieved mean match relative entropy of the temp-CM's mlp7 (`mlp7_re`).
    pub mlp7_re: f64,
    /// Model length M (= cm->clen).
    pub m: i32,
}

/// C: esl_vec_FSum (esl_vectorops.c) — Kahan-compensated summation, exactly as
/// used by esl_vec_FNorm.
#[inline]
fn fsum(v: &[f32]) -> f32 {
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

/// C: esl_vec_FNorm (esl_vectorops.c):
/// ```c
///   sum = esl_vec_FSum(vec, n);
///   if (sum != 0.0) for (x=0;x<n;x++) vec[x] /= sum;
///   else            for (x=0;x<n;x++) vec[x] = 1. / (float) n;
/// ```
#[inline]
fn fnorm(v: &mut [f32]) {
    let sum = fsum(v);
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

/// C: esl_vec_FArgMax (esl_vectorops.c):
/// ```c
///   int best = 0;
///   for (i = 1; i < n; i++) if (vec[i] > vec[best]) best = i;
///   return best;
/// ```
/// Strict `>`, so on ties the FIRST maximal index wins.
#[inline]
fn fargmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// C: esl_vec_FAddScaled(dst, src, scale, n): `dst[i] += src[i]*scale`.
#[inline]
fn faddscaled(dst: &mut [f32], src: &[f32], scale: f32) {
    for i in 0..dst.len() {
        dst[i] += src[i] * scale;
    }
}

/// Build the *temporary* CM (`acm`) exactly as C's `build_and_calibrate_p7_filter`
/// does for its emission-donor CM: the identical `build_model` /
/// `set_effective_seqnumber` / `parameterize` sequence used by the cmbuild bin's
/// `build_one`, EXCEPT the entropy-weight target is `fhmm_re` with the
/// `pretend-model-is-an-HMM` flag TRUE, min = --eminseq (0.1), max = nseq.
///
/// The MSA is taken in its ORIGINAL (as-read) state; this function performs the
/// same deterministic weight/mark/count pipeline as `build_one`, so callers must
/// pass a fresh copy of the MSA (or one whose weighting is idempotent). This
/// mirrors C, whose temp-CM `build_model` runs on the already-weighted `msa`
/// (weighting is deterministic, so the result is identical).
///
/// Returns `(acm, neff, mlp7_re)`.
fn build_temp_cm(
    msa: &mut EslMsa,
    abc: &EslAlphabet,
    fhmm_re: f64,
    knobs: &crate::msaweight::BuildKnobs,
    fullmat: Option<&crate::rsearch::FullMat>,
) -> (CM, f64, f64) {
    let alen = msa.alen as usize;

    // --- check_and_clean_msa: strip SS for --noss, then clean SS_cons (build_one) ---
    // C strips SS in check_and_clean_msa on the SAME msa object later reused by the
    // temp build_model, so the temp CM must see the stripped SS too.
    if knobs.noss {
        msa.ss_cons = Some(".".repeat(alen));
    }
    {
        let mut ss: Vec<u8> = msa.ss_cons.as_ref().expect("no SS_cons").as_bytes().to_vec();
        ss.resize(alen, b'.');
        if !mm::clean_cs(&mut ss, alen) {
            panic!("Failed to parse consensus structure annotation");
        }
        msa.ss_cons = Some(String::from_utf8_lossy(&ss).into_owned());
    }

    // --- check_and_clean_msa: --rsearch degeneracy resolution (cmbuild.c:1416).
    // C's temp CM (build_and_calibrate_p7_filter, cmbuild.c:2408) reuses the same
    // already-resolved msa; here we resolve the pristine clone identically
    // (deterministic → same result as the main build's resolve).
    if let Some(fm) = fullmat {
        crate::rsearch::ribosum_msa_resolve_degeneracies(fm, msa, abc);
    }

    // --- set_relative_weights (build_one) ---
    // C set_relative_weights: ignore_rf = !--hand (RF-defined consensus cols under --hand).
    crate::msaweight::apply_weights(msa, knobs.wscheme, knobs.wid, !knobs.hand);

    // --- mark_fragments / check_fragments (build_one, cmbuild.c:995-1000) ---
    // C never re-marks in build_and_calibrate_p7_filter: it reuses the msa that
    // process_build_workunit already processed once. Under --fraggiven that msa
    // was validated (check_fragments) but NOT modified, so it still carries only
    // the given ~ annotation; skip inference here to stay faithful. Otherwise the
    // pristine clone was already span-marked, and re-marking is idempotent.
    if !knobs.fraggiven {
        mm::mark_fragments(msa, knobs.fragthresh);
    }

    // --- build_model (build_one) ---
    let use_rf = knobs.hand;
    let use_wts = !use_rf && !knobs.v1p0;
    let symfrac = knobs.symfrac;
    let (mut cm, gtr) = mm::hand_modelmaker(msa, abc, use_rf, use_wts, symfrac);

    // C build_model (cmbuild.c:1710-1715): rsearch null = RIBOSUM g, CM_RSEARCHEMIT;
    // else the (possibly --null-overridden) background model.
    if let Some(fm) = fullmat {
        let mut g = [0.0f32; 4];
        g.copy_from_slice(&fm.g[..4]);
        cm.cm_set_null_model(&g);
        cm.flags |= crate::cm::CM_RSEARCHEMIT;
    } else {
        cm.cm_set_null_model(&knobs.null); // C: CMSetNullModel(acm, cfg->null) (--null)
    }
    if !knobs.nobalance {
        cm = crate::cm_rebalance::cm_rebalance(&cm);
    }

    // C determine_pretend_cm_is_hmm: 0 bp AND no force_standard_prior. This path
    // is only reached when use_mlp7_as_filter==FALSE, i.e. determine_pretend..(cm)
    // is FALSE, so this is FALSE too (bp>0, or 0-bp under --noh3pri/etc).
    let pretend_cm_is_hmm =
        cm.cm_count_nodetype(crate::constants::MATP_ND) == 0 && !knobs.force_standard_prior;

    let pri_for_counts = if knobs.use_v0p56_prior {
        crate::prior::prior_v0p56_through_v1p02()
    } else {
        crate::prior::prior_default(pretend_cm_is_hmm)
    };
    let used_el = vec![false; alen + 1];
    let mut trs: Vec<_> = (0..msa.nseq)
        .map(|i| mm::transmogrify(&cm, &gtr, &msa.ax[i], &used_el, alen))
        .collect();
    if pretend_cm_is_hmm {
        for tr in &mut trs {
            mm::cm_parsetree_doctor(&cm, tr);
        }
    }
    for i in 0..msa.nseq {
        mm::parsetree_count(&mut cm, &trs[i], &msa.ax[i], msa.wgt[i] as f32);
    }
    let dbl_e: Vec<Vec<f64>> = (0..cm.m as usize)
        .map(|v| {
            if cm.sttype[v] as i32 == crate::constants::MP_ST {
                cm.e[v].iter().map(|&x| x as f64).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    for i in 0..msa.nseq {
        mm::parsetree_count_only_truncated_mps(
            &mut cm,
            &trs[i],
            &msa.ax[i],
            msa.wgt[i] as f32,
            &dbl_e,
            &pri_for_counts,
        );
    }
    cm.nseq = msa.nseq as i32;
    cm.eff_nseq = msa.nseq as f32;

    if !knobs.iflank && !knobs.v1p0 {
        mm::cm_zero_flanking_insert_counts(&mut cm);
    }
    if !knobs.nodetach {
        mm::cm_find_and_detach_dual_inserts(&mut cm, true, false); // check only
    }

    // el_selfsc/n2/n3 don't affect filter emissions; keep defaults.
    cm.el_selfsc = (0.94f64.ln() * 1.44269504) as f32; // sreLOG2(0.94)
    cm.n2_omega = 0.000015258791;
    cm.n3_omega = 0.000015258791;

    // C uses cfg->pri for the temp CM's cm_EntropyWeight and parameterize:
    // Prior_Default(FALSE), or the v0.56->v1.0.2 prior under --p56/--v1p0.
    let pri = if knobs.use_v0p56_prior {
        crate::prior::prior_v0p56_through_v1p02()
    } else {
        crate::prior::prior_default(false)
    };

    // --- set_effective_seqnumber, but for the filter donor CM ---
    // C cmbuild.c:2411-2417: cm_EntropyWeight(acm, cfg->pri, fhmm_re,
    //   --eminseq (0.1), (--emaxseq used ? : (double) cm->nseq), TRUE, &mlp7_re, &neff);
    // The last argument TRUE says "pretend the model is an HMM for entropy
    // weighting" (== our cm_entropy_weight's pretend_cm_is_hmm=true path).
    let min_neff = knobs.eminseq; // --eminseq (default 0.1)
    let max_neff = knobs.emaxseq.unwrap_or(cm.nseq as f64); // --emaxseq else (double) cm->nseq
    let (mlp7_re, neff) =
        crate::eweight::cm_entropy_weight(&mut cm, &pri, fhmm_re, min_neff, max_neff, true);

    // C cmbuild.c:2417: acm->eff_nseq = neff;  (neff is double, eff_nseq is float)
    cm.eff_nseq = neff as f32;
    // C cmbuild.c:2419: cm_Rescale(acm, acm->eff_nseq / (float) msa->nseq);
    // Faithful transcription: unlike the CM-body path (set_effective_seqnumber,
    // cmbuild.c:2003, which passes the *double* `neff / (float)msa->nseq`), the
    // temp-CM filter path divides the already-f32-truncated `acm->eff_nseq` by
    // `(float)nseq` entirely in f32 (float/float). For the models tested the two
    // formulas yield the same f32 scale, but f32/f32 is what C literally does
    // here, so we match it exactly. (The RNaseP 1-ULP emission divergence was NOT
    // caused by this — see the cm_renormalize Kahan-sum fix in cm.rs::fnorm.)
    let scale = cm.eff_nseq / (msa.nseq as f32);
    crate::eweight::cm_rescale(&mut cm, scale);

    // --- parameterize (build_one) ---
    crate::prior::priorify_cm(&mut cm, &pri);
    // C cmbuild.c:2079: rsearch overwrites emissions from RIBOSUM targets. The temp
    // CM's parameterize() runs this too, so the filter emissions reflect rsearch.
    if let Some(fm) = fullmat {
        crate::rsearch::rsearch_cm_probify_emissions(&mut cm, fm, abc);
    }
    if !knobs.nodetach {
        mm::cm_find_and_detach_dual_inserts(&mut cm, false, true); // detach
    }
    if !knobs.iins {
        mm::flatten_insert_emissions(&mut cm);
    }
    cm.cm_renormalize();

    // --- configure_model(acm, 2): builds W/QDB then acm->cp9 + acm->mlp7.
    // We need cm->W only for mlp7->max_length (irrelevant to emissions/COMPO/CONS);
    // the CP9 emission construction depends only on cm->e/cm->t (unchanged by
    // configure_qdb_and_w). We still run it so the CM is in the same state C's is
    // when build_cp9_hmm runs (harmless for emissions).
    mm::configure_qdb_and_w(&mut cm);

    (cm, neff, mlp7_re)
}

/// Construct the filter HMM's match/insert emission vectors, consensus line and
/// effective sequence number, byte-identical to C cmbuild's default filter build.
///
/// # Arguments
/// * `msa`     - the input MSA in its ORIGINAL (as-read) state; mutated in place
///               by the deterministic weight/mark/clean pipeline (pass a fresh
///               copy). Same object C's temp-CM `build_model` consumes.
/// * `abc`     - RNA alphabet.
/// * `fhmm_re` - stage-A output: `p7_MeanMatchRelativeEntropy(fhmm, bg)`, the
///               entropy-weight target for the temp CM.
/// * `knobs`   - shared build knobs (weighting, symfrac, fragthresh, ...) so the
///               donor CM tracks the same options as the main CM.
pub fn build_filter_emissions(
    msa: &mut EslMsa,
    abc: &EslAlphabet,
    fhmm_re: f64,
    knobs: &crate::msaweight::BuildKnobs,
    fullmat: Option<&crate::rsearch::FullMat>,
) -> FilterEmissions {
    let (cm, neff, mlp7_re) = build_temp_cm(msa, abc, fhmm_re, knobs, fullmat);

    // --- build acm->cp9 (build_cp9_hmm) then acm->mlp7 (cm_cp9_to_p7) ---
    // configure_model(acm,2) -> cm_Configure -> build_cp9_hmm(...) builds the
    // GLOBAL cp9 (no local config: default cmbuild sets no CM_CONFIG_LOCAL),
    // emissions/EL/transitions then CPlan9Renormalize, then CP9Logoddsify.
    // cp9_build_and_configure_global reproduces exactly that (no cp9_sw_config,
    // no EL local ends) — the emission/renormalize path is what we read off.
    let emap = cp9::create_emit_map(&cm);
    let map = cp9::cp9_map_cm2hmm(&cm);
    let psi = cp9::cm_expected_state_occupancy(&cm);
    let tmap = cp9::cm_create_transition_map();
    let cp9hmm = cp9::cp9_build_and_configure_global(&cm, &emap, &map, &psi, &tmap);

    let m = cm.clen as usize; // == cp9hmm.m == fhmm->M
    let k_abc = ALPHABET_SIZE;

    // --- cm_cp9_to_p7 emission portion (cm_p7_modelmaker.c:196-207) ---
    //   for (k=1;k<=clen;k++) esl_vec_FCopy(cp9->mat[k], K, mlp7->mat[k]);
    //   for (k=1;k<=clen;k++) esl_vec_FNorm(mlp7->mat[k], K);
    //   esl_vec_FSet(mlp7->mat[0], K, 0.); mlp7->mat[0][0] = 1.0;
    //   for (k=0;k<=clen;k++) esl_vec_FCopy(cp9->ins[k], K, mlp7->ins[k]);
    //   for (k=0;k<=clen;k++) esl_vec_FNorm(mlp7->ins[k], K);
    let mut mlp7_mat = vec![[0.0f32; ALPHABET_SIZE]; m + 1];
    let mut mlp7_ins = vec![[0.0f32; ALPHABET_SIZE]; m + 1];
    for k in 1..=m {
        mlp7_mat[k] = cp9hmm.mat[k];
    }
    for k in 1..=m {
        fnorm(&mut mlp7_mat[k][..k_abc]);
    }
    mlp7_mat[0] = [0.0; ALPHABET_SIZE];
    mlp7_mat[0][0] = 1.0;
    for k in 0..=m {
        mlp7_ins[k] = cp9hmm.ins[k];
    }
    for k in 0..=m {
        fnorm(&mut mlp7_ins[k][..k_abc]);
    }

    // --- copy mlp7 emissions onto fhmm (cmbuild.c:2431-2440) ---
    //   for (k=1;k<=M;k++) esl_vec_FCopy(acm->mlp7->mat[k], K, fhmm->mat[k]);
    //   for (k=1;k<=M;k++) esl_vec_FNorm(fhmm->mat[k], K);
    //   esl_vec_FSet(fhmm->mat[0], K, 0.); fhmm->mat[0][0] = 1.0;
    //   for (k=0;k<=M;k++) esl_vec_FCopy(acm->mlp7->ins[k], K, fhmm->ins[k]);
    //   for (k=0;k<=M;k++) esl_vec_FNorm(fhmm->ins[k], K);
    let mut mat = vec![[0.0f32; ALPHABET_SIZE]; m + 1];
    let mut ins = vec![[0.0f32; ALPHABET_SIZE]; m + 1];
    for k in 1..=m {
        mat[k] = mlp7_mat[k];
    }
    for k in 1..=m {
        fnorm(&mut mat[k][..k_abc]);
    }
    mat[0] = [0.0; ALPHABET_SIZE];
    mat[0][0] = 1.0;
    for k in 0..=m {
        ins[k] = mlp7_ins[k];
    }
    for k in 0..=m {
        fnorm(&mut ins[k][..k_abc]);
    }

    // --- cm_p7_hmm_SetConsensus(fhmm) (cm_p7_modelmaker.c:557) ---
    //   consensus[0] = ' ';
    //   for (k=1;k<=M;k++) { x = esl_vec_FArgMax(mat[k], K);
    //     consensus[k] = (mat[k][x] >= 0.5) ? toupper(sym[x]) : tolower(sym[x]); }
    let sym = &abc.sym; // RNA: ['A','C','G','U',...], sym[0..3] = A,C,G,U
    let mut consensus = vec![b' '; m + 1];
    for k in 1..=m {
        let x = fargmax(&mat[k][..k_abc]);
        let c = sym[x] as u8;
        consensus[k] = if mat[k][x] >= 0.5 {
            c.to_ascii_uppercase()
        } else {
            c.to_ascii_lowercase()
        };
    }

    FilterEmissions {
        mat,
        ins,
        consensus,
        neff,
        mlp7_re,
        m: m as i32,
    }
}

/// C: p7_hmm_SetComposition (hmmer/src/p7_hmm.c). Sets `hmm->compo[]` from the
/// occupancy-weighted mean of the match/insert emissions. Requires the assembled
/// filter HMM to already hold its (stage-A) transitions, since the occupancy is
/// derived from them.
///
/// ```c
///   p7_hmm_CalculateOccupancy(hmm, mocc, iocc);
///   esl_vec_FSet(compo, K, 0.0);
///   esl_vec_FAddScaled(compo, ins[0], iocc[0], K);
///   for (k=1;k<=M;k++) { esl_vec_FAddScaled(compo, mat[k], mocc[k], K);
///                        esl_vec_FAddScaled(compo, ins[k], iocc[k], K); }
///   esl_vec_FNorm(compo, K);
/// ```
pub fn p7_hmm_set_composition(hmm: &mut P7Profile) {
    let m = hmm.m as usize;
    let k_abc = ALPHABET_SIZE;
    let (mocc, iocc) = p7_hmm_calculate_occupancy(hmm);

    let mut compo = [0.0f32; ALPHABET_SIZE];
    faddscaled(&mut compo[..k_abc], &hmm.ins[0][..k_abc], iocc[0]);
    for k in 1..=m {
        faddscaled(&mut compo[..k_abc], &hmm.mat[k][..k_abc], mocc[k]);
        faddscaled(&mut compo[..k_abc], &hmm.ins[k][..k_abc], iocc[k]);
    }
    fnorm(&mut compo[..k_abc]);
    hmm.compo = compo;
    hmm.flags |= crate::p7_hmm::P7H_COMPO;
}

/// C: p7_hmm_CalculateOccupancy (hmmer/src/p7_hmm.c). `hmm.trans[k]` layout is
/// `[MM,MI,MD,IM,II,DM,DD]` (== p7H_ indices 0..6).
/// ```c
///   mocc[0] = 0.;
///   mocc[1] = t[0][MI] + t[0][MM];               // 1 - B->D_1
///   for (k=2;k<=M;k++)
///     mocc[k] = mocc[k-1]*(t[k-1][MM]+t[k-1][MI]) + (1-mocc[k-1])*t[k-1][DM];
///   iocc[0] = t[0][MI] / t[0][IM];
///   for (k=1;k<=M;k++) iocc[k] = mocc[k]*t[k][MI]/t[k][IM];
/// ```
pub fn p7_hmm_calculate_occupancy(hmm: &P7Profile) -> (Vec<f32>, Vec<f32>) {
    const MM: usize = 0;
    const MI: usize = 1;
    const IM: usize = 3;
    const DM: usize = 5;
    let m = hmm.m as usize;
    let t = &hmm.trans;
    let mut mocc = vec![0.0f32; m + 1];
    let mut iocc = vec![0.0f32; m + 1];

    mocc[0] = 0.0;
    if m >= 1 {
        mocc[1] = t[0][MI] + t[0][MM];
        for k in 2..=m {
            mocc[k] = mocc[k - 1] * (t[k - 1][MM] + t[k - 1][MI])
                + (1.0 - mocc[k - 1]) * t[k - 1][DM];
        }
    }
    iocc[0] = t[0][MI] / t[0][IM];
    for k in 1..=m {
        iocc[k] = mocc[k] * t[k][MI] / t[k][IM];
    }
    (mocc, iocc)
}

/// C: cm_cp9_to_p7() (src/cm_p7_modelmaker.c:118-225) — build the CM's ML p7
/// HMM (`cm->mlp7`) from its (global) CP9 HMM. Used as the filter HMM directly
/// for zero-basepair models (`use_mlp7_as_filter`, cmbuild.c:2331). Transitions
/// come from `cp9->t`, emissions from `cp9->mat`/`cp9->ins`. Returns a P7Profile
/// with everything set EXCEPT the consensus line (cmbuild later calls
/// `cm_p7_hmm_SetConsensus` at threshold 0.5, done by the integrator) and MAP
/// (mlp7 has no alignment map: P7H_MAP stays off).
pub fn cm_cp9_to_p7(cm: &CM) -> P7Profile {
    use crate::p7_hmm::{P7H_CHKSUM, P7H_CS, P7H_RF};
    // p7 transition indices (p7_hmm.h): MM,MI,MD,IM,II,DM,DD = 0..6.
    const P7MM: usize = 0;
    const P7MI: usize = 1;
    const P7MD: usize = 2;
    const P7IM: usize = 3;
    const P7II: usize = 4;
    const P7DM: usize = 5;
    const P7DD: usize = 6;
    // cp9 transition indices (cp9.rs): CTMM=0,CTMI=1,CTMD=2,CTIM=4,CTII=5,CTDM=7,CTDD=9.
    const CTMM: usize = 0;
    const CTMI: usize = 1;
    const CTMD: usize = 2;
    const CTIM: usize = 4;
    const CTII: usize = 5;
    const CTDM: usize = 7;
    const CTDD: usize = 9;

    let m = cm.clen as usize;
    let k_abc = ALPHABET_SIZE;

    // Build the (global) CP9 exactly as the temp-CM emission path does.
    let emap = cp9::create_emit_map(cm);
    let map = cp9::cp9_map_cm2hmm(cm);
    let psi = cp9::cm_expected_state_occupancy(cm);
    let tmap = cp9::cm_create_transition_map();
    let cp9hmm = cp9::cp9_build_and_configure_global(cm, &emap, &map, &psi, &tmap);

    let mut hmm = P7Profile::new(m as i32);
    hmm.name = cm.name.clone();
    hmm.acc = cm.acc.clone();
    hmm.desc = cm.desc.clone();
    hmm.alph = "RNA".to_string();
    // C cm_cp9_to_p7 (cm_p7_modelmaker.c:178-184) copies cm->comlog into mlp7;
    // build_and_calibrate then APPENDS the command again, so the mlp7 filter ends
    // up with the command duplicated (2 COM lines). Reproduce faithfully.
    hmm.comlog = cm.comlog.clone();

    // --- transitions (cm_p7_modelmaker.c:133-159) ---
    for k in 0..=m {
        hmm.trans[k][P7MM] = cp9hmm.t[k][CTMM];
        hmm.trans[k][P7MI] = cp9hmm.t[k][CTMI];
        hmm.trans[k][P7MD] = cp9hmm.t[k][CTMD];
        hmm.trans[k][P7IM] = cp9hmm.t[k][CTIM];
        hmm.trans[k][P7II] = cp9hmm.t[k][CTII];
        hmm.trans[k][P7DM] = cp9hmm.t[k][CTDM];
        hmm.trans[k][P7DD] = cp9hmm.t[k][CTDD];
    }
    // normalize match transitions (t[k][0..3]) for k=1..M
    for k in 1..=m {
        fnorm(&mut hmm.trans[k][0..3]);
    }
    // normalize insert transitions (t[k][3..5]) for k=0..M-1
    for k in 0..m {
        fnorm(&mut hmm.trans[k][3..5]);
    }
    // normalize delete transitions (t[k][5..7]) for k=1..M-1
    for k in 1..m {
        fnorm(&mut hmm.trans[k][5..7]);
    }
    // enforce HMMER conventions
    hmm.trans[m][P7MD] = 0.0;
    fnorm(&mut hmm.trans[m][0..3]);
    hmm.trans[0][P7DM] = 1.0;
    hmm.trans[m][P7DM] = 1.0;
    hmm.trans[0][P7DD] = 0.0;
    hmm.trans[m][P7DD] = 0.0;
    // INFERNAL CP9 convention: node 0 MM transition is begin[1]
    hmm.trans[0][P7MM] = cp9hmm.begin[1];
    fnorm(&mut hmm.trans[0][0..3]);

    // --- match emissions (single FNorm; cm_p7_modelmaker.c:161-166) ---
    for k in 1..=m {
        hmm.mat[k] = cp9hmm.mat[k];
        fnorm(&mut hmm.mat[k][..k_abc]);
    }
    hmm.mat[0] = [0.0; ALPHABET_SIZE];
    hmm.mat[0][0] = 1.0;
    // --- insert emissions (cm_p7_modelmaker.c:168-170) ---
    for k in 0..=m {
        hmm.ins[k] = cp9hmm.ins[k];
        fnorm(&mut hmm.ins[k][..k_abc]);
    }

    // max_length = cm->W (cm_p7_modelmaker.c:172)
    hmm.max_length = cm.w;

    // RF (cm_p7_modelmaker.c:186-190): copy CM's RF if it has one.
    if cm.flags & crate::cm::CM_RF != 0 && (cm.rf.len() as i32) > cm.clen {
        hmm.rf = vec![b' '; m + 2];
        for k in 1..=m {
            hmm.rf[k] = cm.rf[k];
        }
        hmm.flags |= P7H_RF;
    }
    // CS (cm_p7_modelmaker.c:193-201): copy the CM's WUSS consensus structure.
    if let Some(cons) = crate::cm_consensus::create_cm_consensus_full(cm) {
        hmm.cs = vec![b' '; m + 2];
        hmm.cs[0] = b' ';
        for k in 1..=m {
            hmm.cs[k] = cons.cstr[k - 1];
        }
        hmm.flags |= P7H_CS;
    }

    hmm.eff_nseq = cm.eff_nseq;
    hmm.nseq = cm.nseq;
    if cm.flags & crate::cm::CM_CHKSUM != 0 {
        hmm.checksum = cm.checksum;
        hmm.flags |= P7H_CHKSUM;
    }
    // model composition (cm_p7_modelmaker.c:214)
    p7_hmm_set_composition(&mut hmm);

    hmm
}

/// C: cm_p7_hmm_SetConsensus() (src/cm_p7_modelmaker.c:557) — set the p7 HMM's
/// consensus residue line at threshold 0.5 (uppercase if `mat[k][argmax] >= 0.5`).
/// Raises P7H_CONS. Used for the mlp7-as-filter path (0-basepair models), where
/// cmbuild calls this on the final fhmm just like the default path.
pub fn cm_p7_hmm_set_consensus(hmm: &mut P7Profile) {
    const SYM: [u8; 4] = [b'A', b'C', b'G', b'U'];
    let m = hmm.m as usize;
    let mthresh = 0.5f32;
    let mut consensus = vec![b' '; m + 1];
    for k in 1..=m {
        let x = fargmax(&hmm.mat[k][..ALPHABET_SIZE]);
        let c = SYM[x];
        consensus[k] = if hmm.mat[k][x] >= mthresh {
            c.to_ascii_uppercase()
        } else {
            c.to_ascii_lowercase()
        };
    }
    hmm.consensus = consensus;
    hmm.flags |= crate::p7_hmm::P7H_CONS;
}

/// C: p7_hmm_SetConsensus() (hmmer/src/p7_hmm.c) — the *HMMER* consensus setter
/// that `cm_cp9_to_p7` invokes (cm_p7_modelmaker.c:189) when building `cm->mlp7`.
/// Differs from `cm_p7_hmm_set_consensus` only in the case-threshold: HMMER uses
/// `mthresh = 0.9` for RNA/DNA (0.5 for amino). `sq == NULL` here (no reference
/// digital sequence), so `x = esl_vec_FArgMax(mat[k])`. Raises P7H_CONS.
/// ```c
/// else if (hmm->abc->type == eslRNA)   mthresh = 0.9;
/// hmm->consensus[0] = ' ';
/// for (k=1;k<=M;k++) { x = esl_vec_FArgMax(hmm->mat[k], K);
///   hmm->consensus[k] = (mat[k][x] >= mthresh) ? toupper(sym[x]) : tolower(sym[x]); }
/// ```
pub fn p7_hmm_set_consensus(hmm: &mut P7Profile) {
    const SYM: [u8; 4] = [b'A', b'C', b'G', b'U'];
    let m = hmm.m as usize;
    // RNA alphabet: mthresh = 0.9 (HMMER p7_hmm.c). All infernox CMs are RNA.
    let mthresh = 0.9f32;
    let mut consensus = vec![b' '; m + 1];
    for k in 1..=m {
        let x = fargmax(&hmm.mat[k][..ALPHABET_SIZE]);
        let c = SYM[x];
        consensus[k] = if hmm.mat[k][x] >= mthresh {
            c.to_ascii_uppercase()
        } else {
            c.to_ascii_lowercase()
        };
    }
    hmm.consensus = consensus;
    hmm.flags |= crate::p7_hmm::P7H_CONS;
}
