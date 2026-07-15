//! cm_search — reusable library entry point for the byte-parity cmsearch
//! pipeline.
//!
//! The orchestration formerly inlined in `bin/cmsearch.rs::main` lives here so
//! other crates (e.g. tRNAscan-SE) can run the exact same faithful pipeline
//! in-process instead of shelling out to a `cmsearch` subprocess. The binary is
//! now a thin CLI wrapper around [`FaithfulSearcher`].
//!
//! Faithful two-pass cm_Pipeline driver assembled from byte-verified stage
//! functions for the default (non-truncated, non-hmmonly) search:
//!   window loop (ReadWindow: CM_MAX_RESIDUE_COUNT, maxW overlap), both strands
//!     LOOP-1: pli_p7_filter (F1 MSV + F3 lFwd + F3b bias)  [Viterbi off for this Z]
//!             pli_p7_env_def (F4 gFwd + F4b bias + F5 env def)
//!     LOOP-2: pli_cyk_env_filter (F6 CYK, tau=ftau)  → refine envelope
//!             pli_final_stage    (F7 Inside + null3, tau=tau) → hits, pvalue
//!   post: ComputeEvalues → SortForOverlapRemoval → RemoveOrMarkOverlaps
//!         → SortByEvalue → Threshold(E<=E)

#![allow(clippy::too_many_arguments)]

use crate::cm::CM;
use crate::cm_file::cm_file_read_global;
use crate::cm_pipeline::{
    bias_filter_score, build_bias_filter, build_forward_filter, build_msv_filter,
    f3_filter_sequence, p7_bg_null_one, BiasFilter, ForwardFilter, MsvFilter,
    CM_MAX_RESIDUE_COUNT,
};
use crate::cp9::{
    cm_configure_scores, cm_create_transition_map, cm_expected_state_occupancy,
    cp9_build_and_configure, cp9_build_and_configure_global, cp9_build_and_configure_trunc,
    cp9_build_and_configure_trunc_local,
    cp9_iterate_seq2bands, cp9_map_cm2hmm, cp9_seq2bands_trunc, cp9_forward, cp9_backward,
    create_emit_map as cp9_create_emit_map, esl_exp_surv, fast_cyk_scan_hb, fast_finside_scan_hb,
    remove_overlaps_greedy, CP9Bands, CP9, CP9Map, Cp9PostMx,
    DEFAULT_CP9BANDS_THRESH1, DEFAULT_CP9BANDS_THRESH2,
};
use crate::evalue::ExpParams;
use crate::p7_generic::{
    build_glocal_profile, p7_domaindef_glocal, p7_flogsum, p7_gbackward, p7_gforward,
    reconfig_length, GlocalProfile, P7Gmx,
};
use crate::easel::alphabet::EslAlphabet;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Build a 1-based sentinel-padded digital subsequence `[255, res_1..res_L, 255]`
/// from a hit's aligner dsq slice (`hit_dsq[1..=L]` are the emitted residues, matching
/// the parsetree's `emitl`/`emitr` coords). Used to retain the hit subseq for the
/// cmsearch `-A` MSA writer (approach B), mirroring cmalign's dsq layout so the same
/// byte-verified `parsetrees_to_alignment` consumes it identically.
fn sentinel_subdsq(hit_dsq: &[u8], l: i32) -> Vec<u8> {
    let mut d = Vec::with_capacity(l as usize + 2);
    d.push(255u8);
    d.extend_from_slice(&hit_dsq[1..=l as usize]);
    d.push(255u8);
    d
}

/// A genome hit in full-sequence (genome) coordinates. Public result type.
#[derive(Clone, Debug)]
pub struct FaithfulHit {
    /// Index into the `seqs` slice passed to [`FaithfulSearcher::search`].
    pub seq_idx: usize,
    /// True if the hit is on the reverse-complement strand (then start > stop).
    pub in_rc: bool,
    /// Genome coordinate (1-based); for `in_rc`, start > stop.
    pub start: i64,
    pub stop: i64,
    pub score: f32,
    /// null3 correction reported in tblout `bias` column (C: hit->bias).
    pub bias: f32,
    /// tblout `mdl from` (C: ad->cfrom_emit).
    pub mdl_from: i32,
    /// tblout `mdl to` (C: ad->cto_emit).
    pub mdl_to: i32,
    pub gc: f64,
    pub pvalue: f64,
    pub evalue: f64,
    /// tblout `trunc` column: "no" / "5'" / "3'" / "5'&3'".
    pub trunc: String,
    /// tblout `pass` column (pipeline pass index the hit was found in).
    pub pass_idx: i32,
    /// C `hit->hmmonly`: the hit came from the HMM-only pipeline (tblout `mdl`
    /// column = "hmm" instead of "cm"; E-value uses the p7 LF exp-tail).
    pub hmmonly: bool,
    /// Per-hit alignment display (C cm_alidisplay). Populated ONLY on the
    /// `-g --nohmm` path; `None` on the default local path.
    pub alignment: Option<crate::cm_alidisplay::CmAliDisplay>,
}

/// Per-model result produced by the memory-bounded batched flat-pool driver
/// [`FaithfulSearcher::search_many_batched`]. Carries the small CM header fields a
/// caller typically needs for output (name/acc/desc/consensus length) *plus* the
/// model's reported hits, so the caller does not have to keep the (heavy) searcher
/// resident to read them back. These four fields are copied verbatim from the same
/// `CM` a per-model `FaithfulSearcher::search` would expose via [`FaithfulSearcher::cm`],
/// so consuming them is byte-identical to the per-model path.
#[derive(Clone, Debug)]
pub struct ModelResult {
    /// CM `NAME` header field (== `searcher.cm().name`).
    pub name: String,
    /// CM `ACC` header field, if present (== `searcher.cm().acc`).
    pub acc: Option<String>,
    /// CM `DESC` header field, if present (== `searcher.cm().desc`).
    pub desc: Option<String>,
    /// CM `CLEN` (consensus length) header field (== `searcher.cm().clen`).
    pub clen: i32,
    /// Reported hits for this model, byte-identical to `searcher.search(seqs, cfg)`.
    pub hits: Vec<FaithfulHit>,
}

/// Internal working hit (carries the `removed` flag used by overlap removal).
#[derive(Clone, Debug)]
struct Hit {
    seq_idx: usize,
    in_rc: bool,
    start: i64,
    stop: i64,
    score: f32,
    bias: f32,
    mdl_from: i32,
    mdl_to: i32,
    gc: f64,
    pvalue: f64,
    evalue: f64,
    trunc: String,
    pass_idx: i32,
    hmmonly: bool,
    removed: bool,
    alignment: Option<crate::cm_alidisplay::CmAliDisplay>,
}

/// C `--cut_ga` / `--cut_tc` / `--cut_nc`: use the model's curated GA (gathering) /
/// TC (trusted) / NC (noise) bit-score cutoff — read from the CM header GA/TC/NC
/// lines — as the reporting AND inclusion threshold. This is the Rfam-recommended,
/// per-model way to threshold ncRNA hits (e.g. Bakta runs `cmscan --cut_tc`), as
/// opposed to a single global E-value. When set (and the model actually carries the
/// requested cutoff), it overrides `e_report` / `t_cutoff`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelCutoff {
    Ga,
    Tc,
    Nc,
}

/// Search-time configuration (per-call knobs that don't depend on the model).
#[derive(Clone, Debug)]
pub struct FaithfulConfig {
    /// Search only the top (given) strand, not the reverse complement.
    pub toponly: bool,
    /// C `--bottomonly`: search only the bottom (reverse-complement) strand.
    /// C (cm_pipeline.c:235-236): `do_top = !bottomonly`, `do_bot = !toponly`.
    /// With either `toponly` or `bottomonly` set, the search space size `Z` is
    /// NOT doubled (cmsearch.c:2456-2459 only doubles when both are FALSE).
    pub bottomonly: bool,
    /// Report hits with E-value <= `e_report` (C: cmsearch default 10.0).
    pub e_report: f64,
    /// Global (glocal) CM configuration (C: `-g`). Requires `nohmm`.
    pub global: bool,
    /// Skip all HMM filters; run the CM directly (QDB CYK filter + QDB Inside)
    /// (C: `--nohmm`). Only supported together with `global` (the
    /// `-g --nohmm --toponly --notrunc` mode used by tRNAscan-SE Phase-II).
    pub nohmm: bool,
    /// Turn ALL filters off (C: `--max`): no HMM filters, no CYK filter; the
    /// whole window goes straight to a NON-banded (full d-range) global Inside
    /// final stage. Only meaningful with `global`. Used by tRNAscan-SE's
    /// pseudogene NS rescore (`-g --max --toponly --notrunc -T 0`).
    pub max: bool,
    /// Mid-level filtering (C: `--mid`): skip SSV(F1) & Viterbi(F2) HMM filters,
    /// keep Forward/envelope/CYK filters at P<=`Fmid`, run the final stage
    /// HMM-banded — all in GLOBAL config. Only meaningful with `global`. Used by
    /// tRNAscan-SE's isotype `cmscan -g --mid --fmt 2` scan.
    pub mid: bool,
    /// C `--rfam` (cm_pipeline.c:479-490): Rfam-scan filter preset — F1=0.06,
    /// F2=F2b=0.02 (Viterbi + Viterbi-bias ON), F3..F5=0.0002, F6=0.0001, identical to
    /// the >= 20 Gb default tier. Independent of Z. Mutually exclusive with
    /// --max/--nohmm/--mid (which take precedence and turn Viterbi off).
    pub rfam: bool,
    /// C `--cyk` (cm_pipeline.c:696): clear CM_SEARCH_INSIDE so the FINAL CM search
    /// stage uses scanning CYK (not Inside) as the scorer. Changes the reported hit
    /// score, its E-value (CYK exp-tail EXP_CM_LC/GC instead of Inside LI/GI), and
    /// which hits pass inclusion. The hit alignment is unaffected (still optacc+PP in
    /// the normal banded case, since `--acyk` is separate). Applies to all passes.
    pub cyk: bool,
    /// C `-T <x>`: report hits by minimum bit score `x` instead of by E-value.
    /// `None` = default E-value reporting (E <= `e_report`).
    pub t_cutoff: Option<f32>,
    /// C `--cut_ga`/`--cut_tc`/`--cut_nc`: report+include hits scoring >= the model's
    /// curated GA/TC/NC bit-score (from the CM header). Takes precedence over
    /// `t_cutoff`/`e_report` when the model carries the requested cutoff; if it does
    /// not, we fall back to `t_cutoff` then `e_report`. `None` = no model cutoff.
    pub model_cutoff: Option<ModelCutoff>,
    /// C `--notrunc`: disable truncated (TrCYK) alignment passes. Default `true`
    /// for the LIBRARY (existing cm_search callers unaffected); the cmsearch
    /// binary flips it to `false` (truncation ON) to match C's `cmsearch` default.
    /// Truncated passes run only in GLOBAL config (`global`), non-max/non-mid.
    pub notrunc: bool,
    /// C `--5trunc` / `--3trunc` (cm_pipeline.c:351-365), LOCAL-only, mutually
    /// exclusive with each other / `-g` / `--notrunc`: run the STD pass plus only the
    /// 5' (resp. 3') terminal force pass.
    pub trunc5p: bool,
    pub trunc3p: bool,
    /// C `--anytrunc` / `--inttrunc` / `--onlytrunc` (cm_pipeline.c:318-343). These
    /// add the internal PLI_PASS_5P_AND_3P_ANY pass (do_local_envdef=TRUE, LOCAL p7
    /// domain-def) which allows truncation ANYWHERE (internal + terminal), enforcing
    /// neither the first nor final residue. `anytrunc` = STD + terminal FORCE passes +
    /// ANY; `inttrunc` = STD + ANY (no terminal FORCE); `onlytrunc` = ANY only (STD
    /// off). Mutually exclusive; C precedence any>int>only>notrunc>5trunc>3trunc.
    pub anytrunc: bool,
    pub inttrunc: bool,
    pub onlytrunc: bool,
    /// C `--qdb` (cm_pipeline.c:698/707): use QDBs (SMX_QDB2_LOOSE, beta = `--beta`,
    /// default 1e-15) instead of HMM bands in the FINAL Inside round (F7). The F1-F5
    /// HMM filters and the F6 CYK filter are unaffected (still HMM-banded). Default
    /// config only (`--max`/`--nohmm` have their own final-round rules).
    pub qdb: bool,
    /// C `--nonbanded` (cm_pipeline.c:702/706): use the full d-range (SMX_NOQDB) in the
    /// FINAL Inside round instead of HMM bands.
    pub nonbanded: bool,
    /// C `--wcx <x>` (x>=1.25): set W = x * cm->clen for the QDB band calculation
    /// (STAGE 2). Stored here; applied via cm_calc_qdb_bands.
    pub wcx: Option<f64>,
    /// C `--beta <x>` (default 1e-15): tail-loss prob for the final-round QDB band
    /// calculation (SMX_QDB2_LOOSE). STAGE 2 override; None = the built-in 1e-15.
    pub beta: Option<f64>,
    /// C `-Z <x>`: manually set the search-space size (database size) to `<x>`
    /// megabases, overriding the value derived from the target residue count.
    /// C (cmsearch.c:541 / cm_pipeline.c:385): `pli->Z = (int64_t)(x * 1e6)`.
    /// This `Z` is strand-independent (NOT doubled for two-strand search) and is
    /// used both for the search-space-size-dependent filter thresholds
    /// (cm_pipeline.c:497, `Z_Mb = pli->Z/1e6`) and for the final E-value
    /// denominator (`cur_eff_dbsize = (Z/dbsize)*nrandhits`). `None` = derive from
    /// the residue count (default: `2*total_res`, or `total_res` with `toponly`).
    pub z_mb_override: Option<f64>,
    /// C per-stage HMM filter P-value threshold overrides (cm_pipeline.c:575-586):
    /// `--F1` (SSV), `--F3` (Forward), `--F3b` (Forward bias), `--F4` (glocal
    /// Forward), `--F4b` (glocal Forward bias), `--F5` (envelope def). Each, when
    /// `Some`, overrides the Z-dependent tier default, clamped to `ESL_MIN(1.0, x)`.
    /// C only applies these when NOT in `--max`/`--nohmm` mode (and `--F1` also not
    /// in `--mid`); they take effect on the standard + `-g` truncated pipelines.
    pub f1: Option<f64>,
    pub f3: Option<f64>,
    pub f3b: Option<f64>,
    pub f4: Option<f64>,
    pub f4b: Option<f64>,
    pub f5: Option<f64>,
    /// C `--F2` (Viterbi) / `--F2b` (Viterbi bias) / `--F1b` (MSV bias) / `--F5b`
    /// (envelope-def bias) P-value overrides (cm_pipeline.c:575-586). Each, when
    /// `Some`, turns ON the corresponding stage AND sets its threshold (clamped to
    /// `ESL_MIN(1.0, x)`). Applied only when NOT `--max`/`--nohmm` (and F1b/F2 also
    /// not `--mid`).
    pub f2: Option<f64>,
    pub f2b: Option<f64>,
    pub f1b: Option<f64>,
    pub f5b: Option<f64>,
    /// C per-stage on/off overrides (cm_pipeline.c:592-604). `--noF1/2/3/4` skip a
    /// filter stage (do_msv/do_vit/do_fwd/do_gfwd = FALSE); `--noF2b/3b/4b` turn off a
    /// composition-bias sub-filter (do_vitbias/do_fwdbias/do_gfwdbias = FALSE);
    /// `--doF1b`/`--doF5b` turn ON the MSV-bias / env-def-bias sub-filters.
    pub no_f1: bool,
    pub no_f2: bool,
    pub no_f3: bool,
    pub no_f4: bool,
    pub no_f2b: bool,
    pub no_f3b: bool,
    pub no_f4b: bool,
    pub do_f1b: bool,
    pub do_f5b: bool,
    /// C `--F6` (CYK filter P-value). Applied when NOT `--max` (cm_pipeline.c:588),
    /// clamped to `ESL_MIN(1.0, x)`. `None` = tier default (0.0001). Turns on
    /// `do_fcyk` when set. Feeds both the F6 P-test threshold and (via `F6env`) the
    /// CYK envelope-redefinition bit-score cutoff.
    pub f6: Option<f64>,
    /// C `--cykenvx <n>` (default 10): `F6env = ESL_MIN(1.0, F6 * n)`
    /// (cm_pipeline.c:648). `None` = 10.
    pub cykenvx: Option<i64>,
    /// C `--noF6` (cm_pipeline.c:596): `do_fcyk = FALSE` — skip the CYK filter
    /// stage entirely; F5-surviving envelopes pass straight to the final stage.
    pub no_f6: bool,
    /// C `--nocykenv` (cm_pipeline.c:646): `do_fcykenv = FALSE` — the CYK filter
    /// still runs as a P-value gate, but does NOT redefine envelope boundaries.
    pub nocykenv: bool,
    /// C `--tau <x>` (default 5e-6, cm_pipeline.c:651): HMM-band tail-loss prob for
    /// the FINAL round. `None` = 5e-6. Tighter/looser bands change hit coords/scores.
    pub tau: Option<f64>,
    /// C `--ftau <x>` (default 1e-4, cm_pipeline.c:645): HMM-band tail-loss prob for
    /// the CYK FILTER round. `None` = 1e-4.
    pub ftau: Option<f64>,
    /// C `--maxtau <x>` (default 0.05, cm_pipeline.c:240): max tau when tightening
    /// HMM bands to fit the DP matrix size limit. `None` = 0.05.
    pub maxtau: Option<f64>,
    /// C `--FZ <x>` (cm_pipeline.c:497): use `<x>` (Mb) as the search-space size for
    /// selecting the Z-dependent FILTER thresholds ONLY — the E-value Z (search())
    /// is unchanged. `None` = derive the tier from the actual Z. NOTE: the Viterbi
    /// (F2) and Viterbi-bias (F2b) filter stages ARE now ported (see `std_vit_params`
    /// / `f3_filter_sequence`), so the `do_vit` ON tiers (Z_Mb >= 20) and `--rfam` are
    /// byte-faithful. The MSV-bias (F1b, `do_msvbias`) stage is still not ported, but
    /// no default tier or `--rfam` enables it (only the expert `--F1b`/`--doF1b` flags).
    pub fz: Option<f64>,
    /// C `--Fmid <x>` (cmsearch.c:141, default 0.02): with `--mid`, the shared P-value
    /// threshold applied to F3/F3b/F4/F4b/F5/F5b (cm_pipeline.c:476). `None` ⇒ 0.02.
    pub fmid: Option<f64>,
    /// C `--rt1/--rt2/--rt3` (defaults 0.25/0.10/0.20, cm_pipeline.c:309-311): the
    /// glocal domain/envelope-definition region thresholds. `None` = default.
    /// Incompatible with `--max`/`--nohmm` (which skip env-def).
    pub rt1: Option<f64>,
    pub rt2: Option<f64>,
    pub rt3: Option<f64>,
    /// C `--ns <n>` (default 200, cm_pipeline.c:312): number of stochastic-traceback
    /// samples used to resolve a multi-domain region into envelopes. `None` = 200.
    pub ns: Option<i64>,
    /// C `--hmmonly` (cm_pipeline.c NewModel `do_hmmonly_always`): run the HMM-only
    /// pipeline (p7 Forward domain-def) for ALL models, never using the CM. Incompatible
    /// with `-g`. For a model with 0 basepairs, HMM-only fires even without this flag
    /// (unless `nohmmonly`).
    pub hmmonly: bool,
    /// C `--nohmmonly` (`do_hmmonly_never`): never run HMM-only, even for 0-basepair
    /// models. Incompatible with `--hmmmax`.
    pub nohmmonly: bool,
    /// C `--hmmmax` (cm_pipeline.c:613-631): HMM-only with all filters off (do_max):
    /// F1=0.3, F2=1.0, F3=1.0, MSV bias off.
    pub hmmmax: bool,
    /// C `--hmmF1`/`--hmmF2`/`--hmmF3` (defaults 0.02 / 1e-3 / 1e-5): HMM-only
    /// per-stage SSV/Viterbi/Forward P thresholds. `None` = default.
    pub hmm_f1: Option<f64>,
    pub hmm_f2: Option<f64>,
    pub hmm_f3: Option<f64>,
    /// C `--hmmnobias`: turn off the MSV biased-composition filter in HMM-only mode.
    pub hmmnobias: bool,
    /// C `--hmmnonull2`: turn off the null2 score correction in HMM-only mode.
    pub hmmnonull2: bool,
    /// C `--nonull3` (cm_pipeline.c:640 `pli->do_null3 = ... ? FALSE : TRUE`, and
    /// :692 clears CM_SEARCH_NULL3 from `fcyk_cm_search_opts`): turn OFF the NULL3
    /// post-hoc biased-composition score correction. When `true` (do_null3 FALSE),
    /// neither the F6 CYK filter nor the F7 final CM search subtract the null3
    /// correction, and `hit->bias` is 0. Default `false` (null3 ON).
    pub nonull3: bool,
}

impl Default for FaithfulConfig {
    fn default() -> Self {
        Self {
            toponly: false,
            bottomonly: false,
            e_report: 10.0,
            global: false,
            nohmm: false,
            max: false,
            mid: false,
            rfam: false,
            cyk: false,
            t_cutoff: None,
            model_cutoff: None,
            notrunc: true,
            trunc5p: false,
            trunc3p: false,
            anytrunc: false,
            inttrunc: false,
            onlytrunc: false,
            qdb: false,
            nonbanded: false,
            wcx: None,
            beta: None,
            z_mb_override: None,
            f1: None, f3: None, f3b: None, f4: None, f4b: None, f5: None,
            f2: None, f2b: None, f1b: None, f5b: None,
            no_f1: false, no_f2: false, no_f3: false, no_f4: false,
            no_f2b: false, no_f3b: false, no_f4b: false, do_f1b: false, do_f5b: false,
            f6: None, cykenvx: None, no_f6: false, nocykenv: false,
            tau: None, ftau: None, maxtau: None, fz: None, fmid: None,
            rt1: None, rt2: None, rt3: None, ns: None,
            hmmonly: false, nohmmonly: false, hmmmax: false,
            hmm_f1: None, hmm_f2: None, hmm_f3: None,
            hmmnobias: false, hmmnonull2: false,
            nonull3: false,
        }
    }
}

/// A prepared searcher: builds the p7 filters + CP9 HMM + configured CM scores
/// once, then serves any number of [`FaithfulSearcher::search`] calls. All state
/// is read-only after construction, so `search` parallelises internally.
pub struct FaithfulSearcher {
    cm: CM,
    /// CM configured for GLOBAL (glocal) scoring with QDB bands + integer scores,
    /// used by the `-g --nohmm` path. Built once alongside the local CM.
    cm_global: CM,
    /// LOCAL-config CM with QDB bands + integer scores (finite ibeginsc => local
    /// begins), used by the default (non-`-g`) `--max`/`--nohmm` scan path. Clone of
    /// the localized `cm` plus `build_integer_scores` + `cm_calc_qdb_bands`.
    cm_local_scan: CM,
    /// Global Inside exp-tail params (ECMGI) for nohmm final-stage E-values.
    gi: ExpParams,
    /// Global CYK exp-tail params (ECMGC) for the nohmm CYK filter cutoff.
    gc: ExpParams,
    /// Model consensus display info (C CMConsensus_t), built once from cm_global
    /// for the nohmm per-hit cm_alidisplay.
    cmcons: crate::cm_alidisplay::CmConsensus,
    /// Global-config emit map (C cm->emap) for cm_global, used by the truncated
    /// passes' ParsetreeToCMBounds.
    emap_global: crate::cp9::EmitMap,
    /// GLOBAL truncated-begin penalty arrays (C trp->g_ptyAA), built once.
    trp: crate::cm_trunc::TrPenalties,
    /// Per-state marginal left/right emission log-odds (C cm->lmesc/rmesc), full
    /// augmented-alphabet arrays [v][0..Kp] filled by cm_logoddsify (STEP 0).
    lmesc: Vec<Vec<f32>>,
    rmesc: Vec<Vec<f32>>,
    cp9: CP9,
    map: CP9Map,
    /// Truncated-pass CP9 HMMs (C cm->Rcp9/Lcp9/Tcp9), cloned from the global cp9
    /// and configured per truncation mode. Built once (model-only). Stage 0 of the
    /// HB truncated-search port; consumed by the truncated CP9-band computation.
    rcp9: CP9,
    lcp9: CP9,
    tcp9: CP9,
    /// LOCAL-config truncated-pass CP9 HMMs (C cm->Rcp9/Lcp9/Tcp9 when
    /// CM_CONFIG_LOCAL|HMMLOCAL|HMMEL are set — the default non-`-g` cmsearch).
    /// Identical build to `rcp9/lcp9/tcp9` but with EL local-end states added
    /// (cp9_build_and_configure_trunc_local). Used by the default LOCAL truncated
    /// search passes for the truncated CP9-band computation.
    rcp9_local: CP9,
    lcp9_local: CP9,
    tcp9_local: CP9,
    /// GLOBAL-config CP9 HMM (built from `cm_global`) + its map, for the `-g --mid`
    /// HMM-banded path (which bands + scores the global CM, not the localized one).
    cp9_global: CP9,
    map_global: CP9Map,
    mf: MsvFilter,
    ff: ForwardFilter,
    bf: BiasFilter,
    /// F2 (Viterbi) filter, built once from the p7. Drives the STD-path Viterbi (F2)
    /// and Viterbi-bias (F2b) stages (default Z_Mb >= 20 and `--rfam`).
    vf: crate::p7_vitfilter::VitFilter,
    /// Local Viterbi Gumbel mu/lambda (C p7_evparam[CM_p7_LVMU/LVLAMBDA]).
    lvmu: f64,
    lvlambda: f64,
    /// Local MSV Gumbel mu/lambda (C p7_evparam[CM_p7_LMMU/LMLAMBDA]), for the F1b
    /// MSV composition-bias sub-filter (--doF1b/--F1b).
    lmmu: f64,
    lmlambda: f64,
    gm_proto: GlocalProfile,
    /// Truncated-pass p7 GLOCAL/LOCAL profiles (C cm->Rgm/Lgm/Tgm), configured once
    /// at L=100 and (for Rgm/Lgm) length-reconfigured per window in the truncated
    /// p7 env-def stage. Rgm = 5' trunc (UNIGLOCAL), Lgm = 3' trunc (UNILOCAL),
    /// Tgm = 5'&3' trunc (UNILOCAL). Stage-4 dependency for pli_p7_env_def.
    rgm_proto: GlocalProfile,
    lgm_proto: GlocalProfile,
    tgm_proto: GlocalProfile,
    /// Local Forward exp-tail params (C p7_evparam[CM_p7_LFTAU/LFLAMBDA]); the
    /// truncated env-def passes score with these instead of glocal (GFMU/GFLAMBDA).
    lftau: f64,
    lflambda: f64,
    // model-derived scalars
    maxw: usize,
    size_limit: f32,
    gfmu: f64,
    gflambda: f64,
    ln2: f64,
    li: ExpParams,
    lc: ExpParams,
    cyk_env_cutoff: f32,
    /// GLOBAL CYK envelope-redefinition cutoff (C cm->expA[EXP_CM_GC] + log(F6env)/
    /// -lambda), used by the `-g` truncated/STD F6 CYK env filter (GC exp mode).
    gcyk_env_cutoff: f32,
    // F6 / tau constants
    f6: f64,
    fcyk_tau: f64,
    final_tau: f64,
    maxtau: f64,
    /// Per-pass pipeline accounting (C `CM_PLI_ACCT[NPLI_PASSES]`), accumulated
    /// during `search` for the human-readable statistics summary. Interior-mutable
    /// (atomic) so the parallel `&self` window tasks can tally into it; integer sums
    /// are order-independent, so parallel accumulation is bit-exact regardless of
    /// thread scheduling. Reset at the start of each `search` call.
    acct: PliAccounting,
}

/// Number of pipeline passes (C `NPLI_PASSES`, infernal.h:2104). Index by
/// `PLI_PASS_*`: 0=CM_SUMMED, 1=STD_ANY, 2=5P_ONLY_FORCE, 3=3P_ONLY_FORCE,
/// 4=5P_AND_3P_FORCE, 5=5P_AND_3P_ANY, 6=HMM_ONLY_ANY.
pub const NPLI_PASSES: usize = 7;

/// One pass's accounting counters (C `CM_PLI_ACCT`, infernal.h:2120). Only the
/// fields the non-verbose statistics summary needs are tracked. All are `u64`
/// atomics so `&self` parallel tasks can accumulate; summation is order-independent.
#[derive(Default)]
pub struct PassAcct {
    pub npli_top: AtomicU64,
    pub npli_bot: AtomicU64,
    pub nres_top: AtomicU64,
    pub nres_bot: AtomicU64,
    pub n_past_msv: AtomicU64,
    pub pos_past_msv: AtomicU64,
    pub n_past_msvbias: AtomicU64,
    pub pos_past_msvbias: AtomicU64,
    pub n_past_vit: AtomicU64,
    pub pos_past_vit: AtomicU64,
    pub n_past_vitbias: AtomicU64,
    pub pos_past_vitbias: AtomicU64,
    pub n_past_fwd: AtomicU64,
    pub pos_past_fwd: AtomicU64,
    pub n_past_fwdbias: AtomicU64,
    pub pos_past_fwdbias: AtomicU64,
    pub n_past_gfwd: AtomicU64,
    pub pos_past_gfwd: AtomicU64,
    pub n_past_gfwdbias: AtomicU64,
    pub pos_past_gfwdbias: AtomicU64,
    pub n_past_edef: AtomicU64,
    pub pos_past_edef: AtomicU64,
    pub n_past_edefbias: AtomicU64,
    pub pos_past_edefbias: AtomicU64,
    pub n_past_cyk: AtomicU64,
    pub pos_past_cyk: AtomicU64,
    pub n_output: AtomicU64,
    pub pos_output: AtomicU64,
}

/// Plain-value snapshot of one pass's counters (returned to the caller after a
/// search, for rendering the statistics summary). Mirrors `PassAcct` field-for-field.
#[derive(Default, Clone, Copy)]
pub struct PassAcctSnapshot {
    pub npli_top: u64,
    pub npli_bot: u64,
    pub nres_top: u64,
    pub nres_bot: u64,
    pub n_past_msv: u64,
    pub pos_past_msv: u64,
    pub n_past_msvbias: u64,
    pub pos_past_msvbias: u64,
    pub n_past_vit: u64,
    pub pos_past_vit: u64,
    pub n_past_vitbias: u64,
    pub pos_past_vitbias: u64,
    pub n_past_fwd: u64,
    pub pos_past_fwd: u64,
    pub n_past_fwdbias: u64,
    pub pos_past_fwdbias: u64,
    pub n_past_gfwd: u64,
    pub pos_past_gfwd: u64,
    pub n_past_gfwdbias: u64,
    pub pos_past_gfwdbias: u64,
    pub n_past_edef: u64,
    pub pos_past_edef: u64,
    pub n_past_edefbias: u64,
    pub pos_past_edefbias: u64,
    pub n_past_cyk: u64,
    pub pos_past_cyk: u64,
    pub n_output: u64,
    pub pos_output: u64,
}

/// Statistics-summary metadata for `cm_pli_Statistics`: the `pli->F*` thresholds
/// (shown in the "expected" column), the `do_*` filter on/off flags (which lines to
/// print vs "(off)"), the glocal/local CM label, and the model/seq dimensions.
#[derive(Clone, Copy)]
pub struct PliStats {
    pub nmodels: i64,
    pub nnodes: i64,
    pub nseqs: i64,
    pub f1: f64,
    pub f1b: f64,
    pub f2: f64,
    pub f2b: f64,
    pub f3: f64,
    pub f3b: f64,
    pub f4: f64,
    pub f4b: f64,
    pub f5: f64,
    pub f5b: f64,
    pub f6: f64,
    pub do_msv: bool,
    pub do_msvbias: bool,
    pub do_vit: bool,
    pub do_vitbias: bool,
    pub do_fwd: bool,
    pub do_fwdbias: bool,
    pub do_gfwd: bool,
    pub do_gfwdbias: bool,
    pub do_edef: bool,
    pub do_edefbias: bool,
    pub do_fcyk: bool,
    pub do_glocal_cm: bool,
    pub do_trunc_ends: bool,
    /// C `pli->do_trunc_only` (cm_pipeline.c:335). When set, the footer header uses the
    /// distinct "searched for truncated hits" wording (not "re-searched"), since no STD
    /// pass runs (pli_pass_statistics, cm_pipeline.c:59-62/89-91).
    pub do_trunc_only: bool,
}

/// Per-pass pipeline accounting accumulator (C `pli->acct[NPLI_PASSES]`).
pub struct PliAccounting {
    passes: Vec<PassAcct>,
}

impl Default for PliAccounting {
    fn default() -> Self {
        PliAccounting {
            passes: (0..NPLI_PASSES).map(|_| PassAcct::default()).collect(),
        }
    }
}

impl PliAccounting {
    /// Zero every counter (C `cm_pli_ZeroAccounting` for all passes). Called at the
    /// start of each `search` so a reused searcher does not accumulate across calls.
    fn reset(&self) {
        for p in &self.passes {
            for a in [
                &p.npli_top, &p.npli_bot, &p.nres_top, &p.nres_bot,
                &p.n_past_msv, &p.pos_past_msv, &p.n_past_msvbias, &p.pos_past_msvbias,
                &p.n_past_vit, &p.pos_past_vit,
                &p.n_past_vitbias, &p.pos_past_vitbias,
                &p.n_past_fwd, &p.pos_past_fwd,
                &p.n_past_fwdbias, &p.pos_past_fwdbias, &p.n_past_gfwd, &p.pos_past_gfwd,
                &p.n_past_gfwdbias, &p.pos_past_gfwdbias, &p.n_past_edef, &p.pos_past_edef,
                &p.n_past_edefbias, &p.pos_past_edefbias,
                &p.n_past_cyk, &p.pos_past_cyk, &p.n_output, &p.pos_output,
            ] {
                a.store(0, Ordering::Relaxed);
            }
        }
    }
    /// C 1443-1452: npli/nres accounting for one pass, one chunk, one strand.
    /// `nres` is the residue count charged to this pass (`sq->n` for STD/ANY passes,
    /// `min(maxW, sq->n)` for the FORCE truncated passes).
    fn add_pass_res(&self, pass: usize, in_rc: bool, nres: u64) {
        let p = &self.passes[pass];
        if in_rc {
            p.npli_bot.fetch_add(1, Ordering::Relaxed);
            p.nres_bot.fetch_add(nres, Ordering::Relaxed);
        } else {
            p.npli_top.fetch_add(1, Ordering::Relaxed);
            p.nres_top.fetch_add(nres, Ordering::Relaxed);
        }
    }
    /// Add an F1/F3/F3b tally (from `f3_filter_sequence`) to one pass (C 2638/2765/
    /// 2785/2800-2822 — counted once per pass that runs `pli_p7_filter`).
    fn add_f1f3(&self, pass: usize, a: &crate::cm_pipeline::F1F3Acct) {
        let p = &self.passes[pass];
        p.n_past_msv.fetch_add(a.n_past_msv, Ordering::Relaxed);
        p.n_past_msvbias.fetch_add(a.n_past_msvbias, Ordering::Relaxed);
        p.pos_past_msvbias.fetch_add(a.pos_past_msvbias, Ordering::Relaxed);
        p.pos_past_msv.fetch_add(a.pos_past_msv, Ordering::Relaxed);
        p.n_past_vit.fetch_add(a.n_past_vit, Ordering::Relaxed);
        p.pos_past_vit.fetch_add(a.pos_past_vit, Ordering::Relaxed);
        p.n_past_vitbias.fetch_add(a.n_past_vitbias, Ordering::Relaxed);
        p.pos_past_vitbias.fetch_add(a.pos_past_vitbias, Ordering::Relaxed);
        p.n_past_fwd.fetch_add(a.n_past_fwd, Ordering::Relaxed);
        p.pos_past_fwd.fetch_add(a.pos_past_fwd, Ordering::Relaxed);
        p.n_past_fwdbias.fetch_add(a.n_past_fwdbias, Ordering::Relaxed);
        p.pos_past_fwdbias.fetch_add(a.pos_past_fwdbias, Ordering::Relaxed);
    }
    /// C 3149-3150: one window survived glocal Forward (F4).
    fn add_gfwd(&self, pass: usize, wlen: u64) {
        let p = &self.passes[pass];
        p.n_past_gfwd.fetch_add(1, Ordering::Relaxed);
        p.pos_past_gfwd.fetch_add(wlen, Ordering::Relaxed);
    }
    /// C 3179-3180: one window survived glocal Forward bias (F4b).
    fn add_gfwdbias(&self, pass: usize, wlen: u64) {
        let p = &self.passes[pass];
        p.n_past_gfwdbias.fetch_add(1, Ordering::Relaxed);
        p.pos_past_gfwdbias.fetch_add(wlen, Ordering::Relaxed);
    }
    /// C 3265-3266: one envelope survived envelope definition (F5).
    fn add_edef(&self, pass: usize, env_len: u64) {
        let p = &self.passes[pass];
        p.n_past_edef.fetch_add(1, Ordering::Relaxed);
        p.pos_past_edef.fetch_add(env_len, Ordering::Relaxed);
    }
    /// C 3276-3277: one envelope survived the envelope composition-bias filter (F5b).
    fn add_edefbias(&self, pass: usize, env_len: u64) {
        let p = &self.passes[pass];
        p.n_past_edefbias.fetch_add(1, Ordering::Relaxed);
        p.pos_past_edefbias.fetch_add(env_len, Ordering::Relaxed);
    }
    /// C 3453-3454: one envelope survived the CM CYK filter (F6).
    fn add_cyk(&self, pass: usize, env_len: u64) {
        let p = &self.passes[pass];
        p.n_past_cyk.fetch_add(1, Ordering::Relaxed);
        p.pos_past_cyk.fetch_add(env_len, Ordering::Relaxed);
    }
    /// Snapshot one pass's counters as plain values.
    fn snapshot(&self, pass: usize) -> PassAcctSnapshot {
        let p = &self.passes[pass];
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        PassAcctSnapshot {
            npli_top: g(&p.npli_top), npli_bot: g(&p.npli_bot),
            nres_top: g(&p.nres_top), nres_bot: g(&p.nres_bot),
            n_past_msv: g(&p.n_past_msv), pos_past_msv: g(&p.pos_past_msv),
            n_past_msvbias: g(&p.n_past_msvbias), pos_past_msvbias: g(&p.pos_past_msvbias),
            n_past_vit: g(&p.n_past_vit), pos_past_vit: g(&p.pos_past_vit),
            n_past_vitbias: g(&p.n_past_vitbias), pos_past_vitbias: g(&p.pos_past_vitbias),
            n_past_fwd: g(&p.n_past_fwd), pos_past_fwd: g(&p.pos_past_fwd),
            n_past_fwdbias: g(&p.n_past_fwdbias), pos_past_fwdbias: g(&p.pos_past_fwdbias),
            n_past_gfwd: g(&p.n_past_gfwd), pos_past_gfwd: g(&p.pos_past_gfwd),
            n_past_gfwdbias: g(&p.n_past_gfwdbias), pos_past_gfwdbias: g(&p.pos_past_gfwdbias),
            n_past_edef: g(&p.n_past_edef), pos_past_edef: g(&p.pos_past_edef),
            n_past_edefbias: g(&p.n_past_edefbias), pos_past_edefbias: g(&p.pos_past_edefbias),
            n_past_cyk: g(&p.n_past_cyk), pos_past_cyk: g(&p.pos_past_cyk),
            n_output: g(&p.n_output), pos_output: g(&p.pos_output),
        }
    }
}

/// Apply the C command-line per-stage HMM filter P-value overrides
/// (`--F1/--F3/--F3b/--F4/--F4b/--F5`) on top of the Z-dependent tier defaults.
///
/// C cm_pipeline.c:574-586: `--F3..--F5` are applied only when NOT `--max` and NOT
/// `--nohmm`; `--F1` additionally requires NOT `--mid`. Each override is clamped to
/// `ESL_MIN(1.0, x)`. Returns the (possibly overridden) `(f1, f3, f3b, f4, f4b, f5)`.
#[allow(clippy::too_many_arguments)]
fn apply_filter_overrides(
    cfg: &FaithfulConfig,
    do_max: bool,
    nohmm: bool,
    do_mid: bool,
    f1: f64,
    f3: f64,
    f3b: f64,
    f4: f64,
    f4b: f64,
    f5: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let clamp = |x: f64| x.min(1.0);
    let apply_hmm = !do_max && !nohmm; // cm_pipeline.c:580
    let apply_f1 = apply_hmm && !do_mid; // cm_pipeline.c:574 (F1/F2 also exclude --mid)
    let f1 = if apply_f1 { cfg.f1.map(clamp).unwrap_or(f1) } else { f1 };
    let f3 = if apply_hmm { cfg.f3.map(clamp).unwrap_or(f3) } else { f3 };
    let f3b = if apply_hmm { cfg.f3b.map(clamp).unwrap_or(f3b) } else { f3b };
    let f4 = if apply_hmm { cfg.f4.map(clamp).unwrap_or(f4) } else { f4 };
    let f4b = if apply_hmm { cfg.f4b.map(clamp).unwrap_or(f4b) } else { f4b };
    let f5 = if apply_hmm { cfg.f5.map(clamp).unwrap_or(f5) } else { f5 };
    (f1, f3, f3b, f4, f4b, f5)
}

/// Per-model derived pipeline thresholds + final-stage E-value machinery.
/// Extracted so the single-model `search` and the multi-model flat-pool
/// `search_many` share ONE copy of the per-window task body (`run_window_task`)
/// and the finalize path — guaranteeing byte-identical results between them.
/// Which truncation passes the CM pipeline runs, per C cm_pipeline.c:1375-1412.
/// (`Notrunc` = STD only; `Default` = terminal 5P/3P/53; the rest add/replace with
/// the internal 5P_AND_3P_ANY pass.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TruncMode { Notrunc, Default, Trunc5, Trunc3, AnyTrunc, IntTrunc, OnlyTrunc }

struct Thresh {
    f1: f64,
    /// C `pli->do_msvbias`/`pli->F1b`: MSV composition-bias (F1b) sub-filter. OFF by
    /// default (every tier sets do_msvbias=FALSE); turned ON only by `--doF1b`/`--F1b`.
    do_msvbias: bool,
    f1b: f64,
    /// C `pli->do_vit`/`pli->F2`: Viterbi (F2) filter. ON for the default CM pipeline
    /// when Z_Mb >= 20 and for `--rfam`; OFF for Z_Mb < 20, --max/--nohmm/--mid.
    do_vit: bool,
    f2: f64,
    /// C `pli->do_vitbias`/`pli->F2b`: Viterbi composition-bias (F2b) sub-filter.
    do_vitbias: bool,
    f2b: f64,
    /// C `pli->do_fwd`/`do_fwdbias` (F3/F3b, cm_pipeline.c:433-434): local Forward
    /// filter + its composition-bias sub-filter. `--noF3`/`--noF3b` (or --max/--nohmm)
    /// turn them OFF; when off, every window passes that stage.
    do_fwd: bool,
    do_fwdbias: bool,
    /// C `pli->do_gfwd`/`do_gfwdbias` (F4/F4b, cm_pipeline.c:435-436): glocal Forward
    /// filter + bias. `--noF4`/`--noF4b` turn them OFF. NOTE: C has no `if(do_gfwd)`
    /// gate on the F4 filter itself (it always runs); `do_gfwd` only controls the stat
    /// display + counting. `do_gfwdbias` DOES gate the F4b sub-filter (cm_pipeline.c:3146).
    do_gfwd: bool,
    do_gfwdbias: bool,
    /// C `pli->do_edefbias`/`pli->F5b` (cm_pipeline.c:438/586/604): the per-envelope
    /// composition-bias sub-filter. OFF by default; `--doF5b`/`--F5b` turn it ON.
    do_edefbias: bool,
    f5b: f64,
    f3: f64,
    f3b: f64,
    f4: f64,
    f4b: f64,
    f5: f64,
    pli_t: f32,
    cyk_cutoff: f32,
    /// Effective F6 CYK-filter P-value threshold (cm_pipeline.c:589/648): tier
    /// default 0.0001 or the `--F6` override, clamped to `ESL_MIN(1.0, x)`.
    f6: f64,
    /// LOCAL CYK envelope-redefinition bit-score cutoff (`LC.mu + log(F6env)/-LC.lambda`),
    /// used by the standard local pipeline's F6 stage (cm_pipeline.c:3405, EXP_CM_LC).
    cyk_env_cutoff: f32,
    /// GLOBAL CYK envelope-redefinition bit-score cutoff (`GC.mu + log(F6env)/-GC.lambda`),
    /// used by the `-g` std/trunc F6 stages (cm_pipeline.c:3405, EXP_CM_GC).
    gcyk_env_cutoff: f32,
    /// C `--noF6`: `do_fcyk = FALSE` — skip the CYK filter stage.
    do_fcyk: bool,
    /// C `--nocykenv`: `do_fcykenv = FALSE` — CYK filter still gates by P-value but
    /// does not redefine envelope boundaries.
    do_fcykenv: bool,
    /// HMM-band tail-loss probs + max tau (cm_pipeline.c:645/651/240): CYK-filter
    /// round (`--ftau`, default 1e-4), final round (`--tau`, default 5e-6), and the
    /// tightening ceiling (`--maxtau`, default 0.05).
    fcyk_tau: f64,
    final_tau: f64,
    maxtau: f64,
    /// Glocal env-def region thresholds + traceback-sample count (cm_pipeline.c:
    /// 309-312): `--rt1`(0.25), `--rt2`(0.10), `--rt3`(0.20), `--ns`(200).
    rt1: f32,
    rt2: f32,
    rt3: f32,
    ns: usize,
    do_gtrunc: bool,
    /// Default LOCAL truncated search (C `do_trunc_ends` with CM_CONFIG_LOCAL set):
    /// run the STD + 5P/3P/53 passes in LOCAL config. Mutually exclusive with
    /// do_gtrunc (which is the `-g` GLOBAL truncated path).
    do_ltrunc: bool,
    /// C `--notrunc` routed through the STD-only CM pipeline (local or -g).
    do_notrunc_cm: bool,
    /// Which truncation passes to run (C cm_pipeline.c:1375-1412).
    trunc_mode: TruncMode,
    /// C `-g`: CM configured glocal (selects global CM/CP9/exp in the CM pipeline).
    global: bool,
    do_max: bool,
    nohmm: bool,
    do_mid: bool,
    /// C `pli->do_msv` (cm_pipeline.c:429/443/457/472): MSV/SSV window detection.
    /// FALSE for --max/--nohmm/--mid — those tile the chunk into deterministic 2*maxW
    /// windows instead of running MSV (cm_pipeline.c:2618-2633).
    do_msv: bool,
    /// C `--cyk`: final CM search stage uses CYK, not Inside (cm_pipeline.c:696).
    do_cyk: bool,
    /// C `--qdb`/`--nonbanded` (default config): the FINAL Inside round uses QDB
    /// (SMX_QDB2_LOOSE) or the full d-range (SMX_NOQDB) instead of HMM bands; the F6
    /// CYK filter stays HMM-banded. Only one may be set. `--max`/`--nohmm` gate these
    /// out (they have their own final-round rules / windowing).
    do_qdb: bool,
    do_nonbanded: bool,
    e_report: f64,
    t_cutoff: Option<f32>,
    exp_mu: f64,
    exp_lambda: f64,
    ez_final: f64,
    /// C `pli->do_hmmonly_cur`: this model runs the HMM-only pipeline (p7 Forward
    /// domain-def) instead of the CM. When set, `exp_mu`/`exp_lambda`/`ez_final`
    /// carry the p7 LF exp-tail params + nhmmer eZ, and `pli_t` the p7 E2Score.
    hmmonly: bool,
    /// HMM-only p7 filter thresholds (cm_pipeline.c:613-631): F1 (SSV), F2 (Viterbi),
    /// F3 (Forward).
    hmm_f1: f64,
    hmm_f2: f64,
    hmm_f3: f64,
    /// C `do_max_hmmonly` (--hmmmax): all HMM-only filters off (F2/F3 pass everything,
    /// Viterbi/Forward stats lines render "(off)").
    hmm_do_max: bool,
    /// C `pli->do_null2_hmmonly` (cleared by `--hmmnonull2`): null2 dombias in the
    /// hmmonly final stage.
    hmm_do_null2: bool,
    /// C `pli->do_null3` (cm_pipeline.c:640, and CM_SEARCH_NULL3 in fcyk/final
    /// cm_search_opts): apply the NULL3 biased-composition score correction in the
    /// F6 CYK filter and F7 final CM search. TRUE by default; `--nonull3` sets FALSE.
    do_null3: bool,
}

impl FaithfulSearcher {
    /// Build a searcher from a CM file (read in global config, as C does before
    /// localizing). The model must contain a p7 HMM filter.
    pub fn from_cm_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let cm = cm_file_read_global(path)
            .map_err(|e| format!("cannot read covariance model: {:?}", e))?;
        Self::new(cm)
    }

    /// Build a searcher from an already-read (global-config) CM.
    pub fn new(mut cm: CM) -> Result<Self, String> {
        let p7 = match cm.p7.as_ref() {
            Some(p) => p.clone(),
            None => {
                return Err("covariance model has no p7 HMM filter \
                            (build/calibrate the CM before searching)"
                    .to_string())
            }
        };

        // F6 / tau / E-value constants
        let f6: f64 = 0.0001;
        let cykenvx: f64 = 10.0;
        let f6env: f64 = (f6 * cykenvx).min(1.0);
        let fcyk_tau: f64 = 1e-4;
        let final_tau: f64 = 5e-6;
        let maxtau: f64 = 0.05;

        // mxsize_limit = pli_mxsize_limit_from_W(cm->W)  (clamp [256,1024] Mb)
        let size_limit: f32 = {
            let w = cm.w as f32;
            let mut mx = (w - 1000.0) / (3000.0 - 1000.0) * (1024.0 - 256.0) + 256.0;
            mx = mx.min(1024.0);
            mx = mx.max(256.0);
            mx
        };

        // LI (final Inside, local) and LC (fcyk, local) exp-tail params
        let li = cm.exp_params_local.clone();
        let lc = cm.exp_params_local_cyk.clone();

        // cyk_env_cutoff = LC.mu + log(F6env)/(-LC.lambda)
        let cyk_env_cutoff: f32 = (lc.mu + f6env.ln() / (-1.0 * lc.lambda)) as f32;

        // ---- build the GLOBAL CM for the `-g --nohmm` path (before localizing) ----
        // Clone the raw (global-config) CM, configure it globally (no local begins/
        // ends), compute QDB bands (beta1=fbeta=1e-7, beta2=beta=1e-15, betaW=1e-7),
        // and build integer Inside scores. Capture the global exp-tail params.
        let mut cm_global = cm.clone();
        crate::cm_nohmm::cm_configure_scores_global(&mut cm_global);
        // Band betas come from the CM's fields (defaults 1e-7/1e-15/1e-7). C
        // --beta <x> (cmsearch.c:2644 qdbinfo->beta2 = final_beta) overrides the
        // final-round QDB2 tail-loss prob; the bin sets cm.qdb_beta2 = --beta before
        // new(), so the recomputed dmin2/dmax2 use it. beta1 (--fbeta) and beta_W are
        // unchanged by --beta.
        let (qb1, qb2, qbw) = (cm_global.qdb_beta1, cm_global.qdb_beta2, cm_global.w_beta);
        crate::cm_nohmm::cm_calc_qdb_bands(&mut cm_global, qb1, qb2, qbw)
            .map_err(|e| format!("QDB band calculation failed: {e}"))?;
        let gi = cm_global.exp_params.clone(); // ECMGI (global Inside)
        let gc = cm_global.exp_params_global_cyk.clone(); // ECMGC (global CYK)
        // GC-based CYK env cutoff for the `-g` F6 stage (C uses fcyk_cm_exp_mode=GC).
        let gcyk_env_cutoff: f32 = (gc.mu + f6env.ln() / (-1.0 * gc.lambda)) as f32;
        // Model consensus for per-hit cm_alidisplay (built once; model-only).
        let cmcons = crate::cm_alidisplay::create_cm_consensus(&cm_global);

        // Truncated-pass machinery (built once, model-only): emit map, GLOBAL
        // truncation penalties (psi from the global-config CM), marginal emissions.
        let emap_global = crate::cp9::create_emit_map(&cm_global);
        let psi_trunc = crate::cp9::cm_expected_state_occupancy(&cm_global);
        let trp = crate::cm_trunc::TrPenalties::new(&cm_global, &emap_global, &psi_trunc);
        let (lmesc, rmesc) = crate::cm_trunc::marginal_emissions(&cm_global);

        // ---- build the GLOBAL CP9 HMM (from cm_global) for the `-g --mid` path ----
        // cm_global's probabilities (e/t/null) are untouched by scoring/QDB above, so
        // the CP9 built here is the faithful global-config HMM.
        let emap_g = crate::cp9::create_emit_map(&cm_global);
        let map_global = cp9_map_cm2hmm(&cm_global);
        let psi_g = cm_expected_state_occupancy(&cm_global);
        let tmap_g = cm_create_transition_map();
        let cp9_global =
            cp9_build_and_configure_global(&cm_global, &emap_g, &map_global, &psi_g, &tmap_g);

        // ---- build CP9 HMM (from global CM), then configure CM scores for the DP ----
        cm.flags |= (1 << 10) | (1 << 11); // CMH_LOCAL_BEGIN | CMH_LOCAL_END
        let emap = crate::cp9::create_emit_map(&cm);
        let map = cp9_map_cm2hmm(&cm);
        let psi = cm_expected_state_occupancy(&cm);
        let tmap = cm_create_transition_map();
        let cp9 = cp9_build_and_configure(&cm, &emap, &map, &psi, &tmap);
        // Stage 0: build Rcp9/Lcp9/Tcp9 from the SAME global cm.t (C clones cm->cp9
        // before cm_localize touches cm.t; cm_modelconfig.c:287). Must precede
        // cm_configure_scores, which localizes cm.t.
        let (rcp9, lcp9, tcp9) = cp9_build_and_configure_trunc(&cm, &emap, &map, &psi, &tmap);
        // LOCAL-config truncated CP9s (default non-`-g` cmsearch: CM_CONFIG_LOCAL|
        // HMMLOCAL|HMMEL set ⇒ EL local ends added to Rcp9/Lcp9/Tcp9). Built from the
        // same un-localized cm.t as above (C clones cm->cp9 before cm_localize touches
        // cm.t; the EL config reads only structural cm.pend/ndtype). C cm_modelconfig.c:
        // 287-322. This mirrors cmalign's local-config trunc CP9 build.
        let (rcp9_local, lcp9_local, tcp9_local) =
            cp9_build_and_configure_trunc_local(&cm, &emap, &map, &psi, &tmap);
        cm_configure_scores(&mut cm); // localizes cm.t + builds tsc/esc/oesc/beginsc/endsc

        // ---- LOCAL-config scan CM for default (non-`-g`) --max/--nohmm ----
        // `cm` is now localized (finite beginsc/endsc via the localized begin[]/end[]);
        // clone it and add integer scores so the non-HMM integer Inside / float CYK
        // scanner (cm_nohmm::generic_scan) runs with local begins. C computes the QDB
        // bands on the UN-localized model (cm_modelconfig.c:192 CalculateQueryDependent-
        // Bands) BEFORE localizing (:312), so the bands are structural and identical to
        // the ones cm_global already computed on the un-localized structure — copy them
        // rather than recompute on the localized CM (the band-calc expected-length
        // recursion diverges under local begins). W is the same file/cmdline W.
        let mut cm_local_scan = cm.clone();
        crate::cm_nohmm::build_integer_scores(&mut cm_local_scan);
        cm_local_scan.dmin1 = cm_global.dmin1.clone();
        cm_local_scan.dmax1 = cm_global.dmax1.clone();
        cm_local_scan.dmin2 = cm_global.dmin2.clone();
        cm_local_scan.dmax2 = cm_global.dmax2.clone();
        cm_local_scan.qdb_beta1 = cm_global.qdb_beta1;
        cm_local_scan.qdb_beta2 = cm_global.qdb_beta2;

        // ---- p7 filters (built once) ----
        // max_length used for MSV filter reconfig + windowing overlap (= pli->maxW).
        // C cm_pipeline.c:1055: pli->maxW = ESL_MAX(wmult*cm_W, cmult*cm_clen) with
        // wmult=1.0f, cmult=1.25f. pli->maxW is an `int`, so the float MAX is TRUNCATED
        // toward zero on assignment (NOT ceiled) — `as usize` reproduces C's int cast.
        let max_length: usize = (1.0f32 * cm.w as f32).max(1.25f32 * cm.clen as f32) as usize;
        let maxw = max_length;
        let ln2 = std::f64::consts::LN_2;

        let mf = build_msv_filter(&p7, max_length);
        let ff = build_forward_filter(&p7);
        let bf = build_bias_filter(&p7.compo, p7.m as usize);
        // F2 (Viterbi) filter, built once. Used by the STD-path Viterbi/Viterbi-bias
        // stages (default Z_Mb >= 20 tiers and `--rfam`); also matches the HMM-only
        // path's own per-call build. Local Viterbi Gumbel = p7 CM_p7_LVMU/LVLAMBDA.
        let vf = crate::p7_vitfilter::build_vit_filter(&p7);
        let lvmu = p7.evparam.lvmu;
        let lmmu = p7.evparam.lmmu;
        let lmlambda = p7.evparam.lmlambda;
        let lvlambda = p7.evparam.lvlambda;
        let gm_proto = build_glocal_profile(&p7, 100);
        let gfmu = p7.evparam.gfmu;
        let gflambda = p7.evparam.gflambda;
        let lftau = p7.evparam.lftau;
        let lflambda = p7.evparam.lflambda;

        // Truncated-pass p7 profiles (C cm_p7_modelconfig_trunc.c), configured at
        // L=100 exactly as pli_p7_env_def creates them on first use. Rgm/Lgm are
        // clones of the glocal gm reconfigured for 5'/3' truncation; Tgm is a clone
        // of a LOCAL profile reconfigured for 5'&3' truncation.
        let mut rgm_proto = build_glocal_profile(&p7, 100);
        crate::p7_generic::p7_profile_config_5prime_trunc(&mut rgm_proto, 100);
        let mut lgm_proto = build_glocal_profile(&p7, 100);
        crate::p7_generic::p7_profile_config_3prime_trunc(&p7, &mut lgm_proto, 100);
        let mut tgm_proto = crate::p7_generic::build_local_profile(&p7, 100);
        crate::p7_generic::p7_profile_config_5prime_and_3prime_trunc(&mut tgm_proto, 100);

        Ok(Self {
            cm,
            cm_global,
            cm_local_scan,
            gi,
            gc,
            cmcons,
            emap_global,
            trp,
            lmesc,
            rmesc,
            cp9,
            map,
            rcp9,
            lcp9,
            tcp9,
            rcp9_local,
            lcp9_local,
            tcp9_local,
            cp9_global,
            map_global,
            mf,
            ff,
            bf,
            vf,
            lvmu,
            lmmu,
            lmlambda,
            lvlambda,
            gm_proto,
            rgm_proto,
            lgm_proto,
            tgm_proto,
            lftau,
            lflambda,
            maxw,
            size_limit,
            gfmu,
            gflambda,
            ln2,
            li,
            lc,
            cyk_env_cutoff,
            gcyk_env_cutoff,
            f6,
            fcyk_tau,
            final_tau,
            maxtau,
            acct: PliAccounting::default(),
        })
    }

    /// Snapshot the per-pass pipeline accounting after a `search` (C `pli->acct`).
    /// Index by `PLI_PASS_*` (1..=5); pass 0 (CM_SUMMED) is computed by the caller.
    pub fn acct_snapshot(&self, pass: usize) -> PassAcctSnapshot {
        self.acct.snapshot(pass)
    }

    /// Build the "Internal HMM-only pipeline statistics summary" data (C
    /// `pli_hmmonly_pass_statistics`), or `None` if this model does not run HMM-only.
    /// The F2 (Viterbi) filter is not yet wired: the Viterbi window count reuses the
    /// MSV-bias survivors as a placeholder (a known 1-line gap until F2 lands).
    pub fn hmmonly_pass_stats(
        &self,
        seqs: &[&str],
        cfg: &FaithfulConfig,
    ) -> Option<crate::p7_hmmonly::HmmonlyPassStats> {
        let cm_nbp = self
            .cm
            .ndtype
            .iter()
            .filter(|&&t| t as i32 == crate::constants::MATP_ND)
            .count() as i32;
        if !crate::p7_hmmonly::newmodel_do_hmmonly_cur(
            cfg.nohmmonly || cfg.max || cfg.nohmm,
            cfg.global,
            cfg.hmmonly,
            cm_nbp,
        ) {
            return None;
        }
        let fcfg = crate::p7_hmmonly::hmmonly_filter_cfg(
            cfg.hmm_f1.unwrap_or(0.02) as f32,
            cfg.hmm_f2.unwrap_or(1e-3) as f32,
            cfg.hmm_f3.unwrap_or(1e-5) as f32,
            cfg.hmmmax,
            cfg.hmmnonull2,
            cfg.hmmnobias,
        );
        let a = self.acct.snapshot(crate::cm_trunc::PLI_PASS_HMM_ONLY_ANY as usize);
        let p7 = self.cm.p7.as_ref()?;
        Some(crate::p7_hmmonly::HmmonlyPassStats {
            do_hmmonly_always: cfg.hmmonly,
            search_mode: true,
            // C `pli->nmodels` counts models that used the CM pipeline (cm_pipeline.c:1037),
            // = 0 for a pure --hmmonly run → match_cm_spacing FALSE (narrow summary format).
            nmodels: 0,
            nmodels_hmmonly: 1,
            nnodes_hmmonly: p7.m as i64,
            nseqs: seqs.len() as i64,
            nres_searched: (a.nres_top + a.nres_bot) as i64,
            do_bias: fcfg.do_bias,
            do_max: fcfg.do_max,
            f1: fcfg.f1,
            f2: fcfg.f2,
            f3: fcfg.f3,
            n_past_msv: a.n_past_msv as i64,
            pos_past_msv: a.pos_past_msv as i64,
            n_past_msvbias: a.n_past_msv as i64,
            pos_past_msvbias: a.pos_past_msv as i64,
            n_past_vit: a.n_past_vit as i64,
            pos_past_vit: a.pos_past_vit as i64,
            n_past_fwd: a.n_past_fwd as i64,
            pos_past_fwd: a.pos_past_fwd as i64,
            n_output: a.n_output as i64,
            pos_output: a.pos_output as i64,
        })
    }

    /// Summed (`PLI_PASS_CM_SUMMED`) accounting: C `pli_sum_statistics`
    /// (cm_pipeline.c) — sum passes 1..6 (skipping HMM_ONLY) whose nres>0.
    pub fn acct_summed(&self) -> PassAcctSnapshot {
        let mut s = PassAcctSnapshot::default();
        for p in 1..NPLI_PASSES {
            if p == 6 {
                continue; // skip PLI_PASS_HMM_ONLY_ANY
            }
            let a = self.acct.snapshot(p);
            if a.nres_top == 0 && a.nres_bot == 0 {
                continue;
            }
            s.npli_top += a.npli_top; s.npli_bot += a.npli_bot;
            s.nres_top += a.nres_top; s.nres_bot += a.nres_bot;
            s.n_past_msv += a.n_past_msv; s.pos_past_msv += a.pos_past_msv;
            s.n_past_msvbias += a.n_past_msvbias; s.pos_past_msvbias += a.pos_past_msvbias;
            s.n_past_vit += a.n_past_vit; s.pos_past_vit += a.pos_past_vit;
            s.n_past_vitbias += a.n_past_vitbias; s.pos_past_vitbias += a.pos_past_vitbias;
            s.n_past_fwd += a.n_past_fwd; s.pos_past_fwd += a.pos_past_fwd;
            s.n_past_fwdbias += a.n_past_fwdbias; s.pos_past_fwdbias += a.pos_past_fwdbias;
            s.n_past_gfwd += a.n_past_gfwd; s.pos_past_gfwd += a.pos_past_gfwd;
            s.n_past_gfwdbias += a.n_past_gfwdbias; s.pos_past_gfwdbias += a.pos_past_gfwdbias;
            s.n_past_edef += a.n_past_edef; s.pos_past_edef += a.pos_past_edef;
            s.n_past_edefbias += a.n_past_edefbias; s.pos_past_edefbias += a.pos_past_edefbias;
            s.n_past_cyk += a.n_past_cyk; s.pos_past_cyk += a.pos_past_cyk;
            s.n_output += a.n_output; s.pos_output += a.pos_output;
        }
        s
    }

    /// Statistics-summary metadata (C `pli->F*` thresholds + `do_*` filter flags +
    /// model dims) for `cm_pli_Statistics`. Derived from the same per-search
    /// thresholds the pipeline uses. Only the Z<20Mb tier (do_vit/msvbias/vitbias/
    /// edefbias OFF) — the tier infernox supports — is described here.
    pub fn pli_stats(&self, seqs: &[&str], cfg: &FaithfulConfig) -> PliStats {
        let th = self.derive_thresholds(seqs, cfg);
        let std_filters = !cfg.max && !cfg.nohmm; // HMM filters run
        PliStats {
            nmodels: 1,
            nnodes: self.cm.clen as i64,
            nseqs: seqs.len() as i64,
            f1: th.f1, f1b: th.f1b, f2: th.f2, f2b: th.f2b, f3: th.f3, f3b: th.f3b, f4: th.f4, f4b: th.f4b, f5: th.f5, f5b: th.f5b, f6: th.f6,
            do_msv: th.do_msv,
            do_msvbias: th.do_msvbias,
            do_vit: th.do_vit,
            do_vitbias: th.do_vitbias,
            do_fwd: th.do_fwd,
            do_fwdbias: th.do_fwdbias,
            do_gfwd: th.do_gfwd,
            do_gfwdbias: th.do_gfwdbias,
            do_edef: std_filters,
            do_edefbias: th.do_edefbias,
            do_fcyk: th.do_fcyk,
            do_glocal_cm: cfg.global,
            // C `do_trunc_ends` (cmsearch default): truncation runs unless disabled by
            // --max/--nohmm (cm_pipeline.c:448/463). --mid does NOT disable it (line
            // 470-478 leaves do_trunc_ends=TRUE), so BOTH local and `-g` --mid run the
            // 5P/3P/53 passes and the footer reports "re-searched N". --qdb
            // (cm_pipeline.c:743-758) DOES kill all truncation, so the footer reports
            // "re-searched 0" like --max/--nohmm.
            do_trunc_ends: !cfg.notrunc && !cfg.max && !cfg.nohmm && !cfg.qdb,
            // C `do_trunc_only`: no STD pass runs, so the footer uses the "searched"
            // (not "re-searched") wording. Disabled by --max/--nohmm (which kill all
            // truncation and route to the STD-only scan path).
            do_trunc_only: cfg.onlytrunc && !cfg.max && !cfg.nohmm,
        }
    }

    /// The model name (CM `NAME`), for tblout `query name`.
    pub fn model_name(&self) -> &str {
        &self.cm.name
    }

    /// VERIFICATION HARNESS (Stage-1 gate). Compute the truncated CP9 bands for a
    /// window `dsq[i0..=j0]` under pipeline pass `pass_idx`, exactly as C
    /// cp9_Seq2Bands does. Returns (fwd_sc, bck_sc, bands, posterior_mx). `dsq` is
    /// the 1-based digitized full sequence; i0/j0 are 1-based window coords.
    ///
    /// Mirrors C's cp9 selection (hmmband.c:249-255): 5P->Rcp9, 3P->Lcp9,
    /// 5P&3P->Tcp9. Used to cross-check against the C DUMP_CP9FB/DUMP_CP9HD dumps.
    pub fn debug_trunc_bands(
        &self,
        dsq: &[u8],
        i0: i32,
        j0: i32,
        pass_idx: i32,
        tau: f64,
    ) -> (f32, f32, CP9Bands, Cp9PostMx) {
        use crate::cm_trunc::{
            PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE, PLI_PASS_5P_AND_3P_FORCE,
            PLI_PASS_5P_AND_3P_ANY,
        };
        let cp9: &CP9 = match pass_idx {
            x if x == PLI_PASS_5P_ONLY_FORCE => &self.rcp9,
            x if x == PLI_PASS_3P_ONLY_FORCE => &self.lcp9,
            x if x == PLI_PASS_5P_AND_3P_FORCE || x == PLI_PASS_5P_AND_3P_ANY => &self.tcp9,
            _ => &self.cp9,
        };
        let emap = cp9_create_emit_map(&self.cm);
        let do_fwd_scan = !crate::cp9::cm_pli_pass_enforces_first_res(pass_idx);
        let do_bck_scan = !crate::cp9::cm_pli_pass_enforces_final_res(pass_idx);
        let (fsc, _fp, _fmx, _fa) =
            cp9_forward(cp9, dsq, i0 as usize, j0 as usize, do_fwd_scan, false);
        let (bsc, _bp, _bmx, _ba) =
            cp9_backward(cp9, dsq, i0 as usize, j0 as usize, do_bck_scan, false);
        let (bands, pmx) = cp9_seq2bands_trunc(
            &self.cm, cp9, &self.map, &emap, dsq, i0, j0, tau, pass_idx,
            DEFAULT_CP9BANDS_THRESH1, DEFAULT_CP9BANDS_THRESH2,
        );
        (fsc, bsc, bands, pmx)
    }

    /// The bsc[1] tag C prints in DUMP_CP9HD (`cp9->otsc[cp9O_NTRANS+cp9O_BM]`),
    /// per truncated pass. Lets the harness correlate a CP9HD dump line to a pass.
    pub fn debug_trunc_cp9_tag(&self, pass_idx: i32) -> i32 {
        use crate::cm_trunc::{
            PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE, PLI_PASS_5P_AND_3P_FORCE,
            PLI_PASS_5P_AND_3P_ANY,
        };
        let cp9: &CP9 = match pass_idx {
            x if x == PLI_PASS_5P_ONLY_FORCE => &self.rcp9,
            x if x == PLI_PASS_3P_ONLY_FORCE => &self.lcp9,
            x if x == PLI_PASS_5P_AND_3P_FORCE || x == PLI_PASS_5P_AND_3P_ANY => &self.tcp9,
            _ => &self.cp9,
        };
        cp9.bsc[1]
    }

    /// The model accession (CM `ACC`), for tblout `query accession`; `"-"` if none.
    pub fn model_acc(&self) -> &str {
        self.cm.acc.as_deref().unwrap_or("-")
    }

    /// The underlying configured CM (read-only).
    pub fn cm(&self) -> &CM {
        &self.cm
    }

    /// The model's curated cutoff bit-score for `mc` (C GA/TC/NC header lines), or
    /// `None` if the model does not carry that cutoff (flag unset). Lets callers
    /// read the Rfam thresholds the way `cmscan --cut_ga/--cut_tc/--cut_nc` does.
    pub fn model_cutoff(&self, mc: ModelCutoff) -> Option<f32> {
        use crate::cm::{CM_GA, CM_NC, CM_TC};
        match mc {
            ModelCutoff::Ga => (self.cm.flags & CM_GA != 0).then_some(self.cm.ga),
            ModelCutoff::Tc => (self.cm.flags & CM_TC != 0).then_some(self.cm.tc),
            ModelCutoff::Nc => (self.cm.flags & CM_NC != 0).then_some(self.cm.nc),
        }
    }

    /// Resolve the effective bit-score reporting/inclusion cutoff for a config,
    /// mirroring C's precedence: `--cut_ga/tc/nc` (this model's GA/TC/NC) wins when
    /// present; else the explicit `-T` (`t_cutoff`); else `None` (E-value reporting).
    fn resolved_t_cutoff(&self, cfg: &FaithfulConfig) -> Option<f32> {
        match cfg.model_cutoff {
            Some(mc) => self.model_cutoff(mc).or(cfg.t_cutoff),
            None => cfg.t_cutoff,
        }
    }

    /// Run the faithful cmsearch pipeline over `seqs` (each an ASCII residue
    /// string). Returns reported hits (E <= `cfg.e_report`) in C's SortByEvalue
    /// order. `seq_idx` on each hit indexes back into `seqs`.
    pub fn search(&self, seqs: &[&str], cfg: &FaithfulConfig) -> Vec<FaithfulHit> {
        let abc = EslAlphabet::rna();
        // Single-model search shares EXACTLY the same machinery as the multi-model
        // flat pool (`search_many`): derive per-model thresholds, build the (chunk ×
        // strand) window task list, run each task, then finalize (overlap removal +
        // SortByEvalue + reporting threshold). Delegating here (rather than an inline
        // copy) guarantees `search` and `search_many` remain byte-identical.
        // Fresh accounting for this search (C cm_pli_ZeroAccounting at pipeline create).
        self.acct.reset();
        let th = self.derive_thresholds(seqs, cfg);
        let fulls: Vec<Vec<u8>> = seqs.iter().map(|s| digitize_sent(&abc, s)).collect();
        let tasks = self.window_tasks(&fulls, cfg.toponly, cfg.bottomonly);
        let per_task: Vec<Vec<Hit>> = tasks
            .par_iter()
            .map(|&task| self.run_window_task(&fulls, task, &th))
            .collect();
        let all_hits: Vec<Hit> = per_task.into_iter().flatten().collect();
        let reported = self.finalize(all_hits, &th);
        // C cmsearch.c:675-680: per reported/included hit, tally n_output/pos_output
        // into that hit's pass (pass_idx). Done after threshold, from the final list.
        for h in &reported {
            let p = &self.acct.passes[h.pass_idx as usize];
            p.n_output.fetch_add(1, Ordering::Relaxed);
            p.pos_output.fetch_add((h.stop - h.start).unsigned_abs() + 1, Ordering::Relaxed);
        }
        reported
    }

    /// Search MANY models over `seqs` with a single FLAT (model × window × strand)
    /// rayon task pool. Returns one hit vector per input searcher, each byte-identical
    /// to `searchers[i].search(seqs, cfg)`.
    ///
    /// Why a flat pool: a per-model parallel loop (one rayon task per whole model)
    /// leaves the last few giant models (23S/16S rRNA) gating the wall — a single
    /// model's own window-parallelism is shallow (only the ~handful of windows that
    /// survive the p7 filter do heavy CYK), so it can't fill the cores alone. A
    /// per-window loop over ONE model, conversely, starves on small models (few
    /// heavy windows). Unioning every model's windows into ONE pool keeps all cores
    /// busy through both the bulk and the tail. Results are regrouped per model and
    /// finalized exactly as the single-model path, so output is unchanged.
    pub fn search_many(
        searchers: &[FaithfulSearcher],
        seqs: &[&str],
        cfg: &FaithfulConfig,
    ) -> Vec<Vec<FaithfulHit>> {
        if searchers.is_empty() {
            return Vec::new();
        }
        let abc = EslAlphabet::rna();
        let fulls: Vec<Vec<u8>> = seqs.iter().map(|s| digitize_sent(&abc, s)).collect();
        let ths: Vec<Thresh> = searchers
            .iter()
            .map(|s| s.derive_thresholds(seqs, cfg))
            .collect();
        // Flat task list; each entry tags its model index so the worker can pull the
        // right searcher + thresholds. Built in model order, then window order — the
        // same per-model window order the single-model `search` uses.
        let mut flat: Vec<(usize, (usize, usize, usize, usize, bool, usize))> = Vec::new();
        for (mi, s) in searchers.iter().enumerate() {
            for t in s.window_tasks(&fulls, cfg.toponly, cfg.bottomonly) {
                flat.push((mi, t));
            }
        }
        let per_task: Vec<(usize, Vec<Hit>)> = flat
            .par_iter()
            .map(|&(mi, t)| (mi, searchers[mi].run_window_task(&fulls, t, &ths[mi])))
            .collect();
        // Regroup per model (collect preserves flat order → each model's windows land
        // in original order), then finalize each model as `search` does.
        let mut grouped: Vec<Vec<Hit>> = (0..searchers.len()).map(|_| Vec::new()).collect();
        for (mi, hits) in per_task {
            grouped[mi].extend(hits);
        }
        grouped
            .into_iter()
            .enumerate()
            .map(|(mi, hits)| searchers[mi].finalize(hits, &ths[mi]))
            .collect()
    }

    /// MEMORY-BOUNDED streaming flat pool: search `n_models` models over `seqs`
    /// while keeping at most `batch_size` searchers resident at once.
    ///
    /// [`search_many`] gives the best wall-clock (every model's windows share ONE
    /// rayon pool, so the giant-model tail can't gate the wall), but it requires
    /// ALL searchers to be built and resident simultaneously — a large footprint
    /// for a 1000+-model Rfam sweep. This driver processes models in batches:
    ///
    ///   for each batch of up to `batch_size` model indices:
    ///     1. build that batch's searchers in parallel (via `build`),
    ///     2. flat-pool ALL their windows across ALL cores (exactly `search_many`),
    ///     3. finalize + snapshot each model's result, then
    ///     4. DROP the batch's searchers before starting the next batch.
    ///
    /// Peak resident searchers is therefore bounded by (successful builds in) one
    /// batch, not the whole DB — trading throughput for a hard memory cap. Set
    /// `batch_size >= n_models` to recover the all-resident `search_many` behaviour.
    ///
    /// WHEN THIS HELPS vs HURTS (measured): the flat pool wins when the number of
    /// models is small relative to the core count and a few giant models would
    /// otherwise each gate a core (its original design target — pre-built searchers,
    /// no build cost on the critical path). It LOSES for large many-model DBs,
    /// because each batch must build every searcher BEFORE any window search can
    /// start (windows can't be enumerated without a built searcher): the slow
    /// giant-model build (23S/16S QDB+CP9 construction) then gates the whole batch's
    /// search phase. A per-model driver that interleaves each model's build with
    /// other models' searches (e.g. `models.par_iter().map(|m| build(m).search())`)
    /// keeps cores busier and was measured 24-38 % FASTER at 4/8/16 threads on a
    /// 144-model DB (and ~1 % faster, ~0.8 GB lighter, on a full 1104-model MG1655
    /// sweep). Prefer that per-model form for 100+-model sweeps; use this driver for
    /// few-model / already-resident-searcher workloads.
    ///
    /// `build(i)` returns the searcher for model `i`, or `None` to skip it (parse
    /// failure / uncalibrated model / caught panic — the caller owns that policy
    /// and any warning). It is called at most once per model, possibly from
    /// several worker threads at once, so it must be `Sync`. The returned vector
    /// is indexed by model: `result[i]` is `Some(ModelResult)` when `build(i)`
    /// produced a searcher, else `None`.
    ///
    /// BYTE-PARITY (by construction): within a batch this calls the exact same
    /// [`search_many`] machinery, whose per-model output equals the single-model
    /// [`FaithfulSearcher::search`]. Crucially the per-model filter thresholds and
    /// final E-value machinery ([`Thresh`]) depend ONLY on `seqs`/`cfg` — never on
    /// which other models share the batch — so a model's result is independent of
    /// its batch assignment. Grouping models into batches (in any order, of any
    /// size) thus changes nothing about any model's reported hits. The four
    /// [`ModelResult`] header fields are copied from the same `CM` the per-model
    /// path exposes. Hence the emitted hits are identical for every `batch_size`.
    pub fn search_many_batched<F>(
        n_models: usize,
        batch_size: usize,
        seqs: &[&str],
        cfg: &FaithfulConfig,
        build: F,
    ) -> Vec<Option<ModelResult>>
    where
        F: Fn(usize) -> Option<FaithfulSearcher> + Sync,
    {
        let batch_size = batch_size.max(1);
        let mut out: Vec<Option<ModelResult>> = (0..n_models).map(|_| None).collect();
        let mut start = 0usize;
        while start < n_models {
            let end = (start + batch_size).min(n_models);
            // 1. Build this batch's searchers in parallel. Keep each model's global
            //    index so we can scatter results back into `out` and drop misses.
            let built: Vec<(usize, FaithfulSearcher)> = (start..end)
                .into_par_iter()
                .filter_map(|i| build(i).map(|s| (i, s)))
                .collect();
            if !built.is_empty() {
                // Snapshot header fields BEFORE searching (borrow, no clone of the CM
                // itself beyond these small fields), then flat-pool the batch.
                let metas: Vec<(usize, String, Option<String>, Option<String>, i32)> = built
                    .iter()
                    .map(|(gi, s)| {
                        let cm = s.cm();
                        (*gi, cm.name.clone(), cm.acc.clone(), cm.desc.clone(), cm.clen)
                    })
                    .collect();
                let searchers: Vec<FaithfulSearcher> =
                    built.into_iter().map(|(_, s)| s).collect();
                // 2. Flat pool across the whole batch (identical to `search_many`).
                let per_model = Self::search_many(&searchers, seqs, cfg);
                // 3. Snapshot each model's result at its global index.
                for ((gi, name, acc, desc, clen), hits) in metas.into_iter().zip(per_model) {
                    out[gi] = Some(ModelResult { name, acc, desc, clen, hits });
                }
                // 4. `searchers` drops here → batch freed before the next iteration.
            }
            start = end;
        }
        out
    }

    /// Derive the Z-dependent filter thresholds + per-model final-stage E-value
    /// machinery for `seqs`/`cfg`. Reproduces the setup block of `search` so the
    /// multi-model flat-pool driver can precompute one `Thresh` per model.
    fn derive_thresholds(&self, seqs: &[&str], cfg: &FaithfulConfig) -> Thresh {
        let total_res: i64 = seqs.iter().map(|s| s.len() as i64).sum();
        // C `-Z <x>`: manual search-space size in Mb (see search()).
        let z: f64 = match cfg.z_mb_override {
            Some(mb) => ((mb * 1_000_000.0) as i64) as f64,
            None => if cfg.toponly || cfg.bottomonly { total_res as f64 } else { (total_res * 2) as f64 },
        };
        // C `--FZ <x>` (cm_pipeline.c:497): the FILTER-threshold tier is selected from
        // <x> Mb when given, else from the actual Z. E-value Z (above) is unaffected.
        let z_mb = match cfg.fz {
            Some(fz) => fz,
            None => z / 1_000_000.0,
        };
        let smallx1 = 1e-6_f64;
        // Hoisted above the tier block so the Viterbi (F2/F2b) tier logic can see them
        // (C forces do_vit=do_vitbias=FALSE under --max/--nohmm/--mid).
        let nohmm = cfg.nohmm;
        let do_max = cfg.max;
        let do_mid = cfg.global && cfg.mid;
        let (f1, f3, f3b, f4, f4b, f5): (f64, f64, f64, f64, f64, f64);
        // C base defaults (cm_pipeline.c:431-432): do_vit = do_vitbias = TRUE with the
        // per-tier F2/F2b below. For Z_Mb < 20 the default `else` branch turns them off
        // (lines 545/554); --max/--nohmm/--mid also turn them off (forced below).
        let (mut do_vit, mut do_vitbias): (bool, bool);
        let (mut f2, mut f2b): (f64, f64);
        // C cm_pipeline.c:470-478: `--mid` preset — a top-level filter-strategy branch
        // (mutually exclusive with --max/--nohmm/--rfam and the Z-dependent default
        // tier). It turns OFF MSV(F1)/Viterbi(F2) and sets ONE shared P-value threshold
        // for ALL remaining HMM stages: F3 = F3b = F4 = F4b = F5 = F5b = --Fmid (default
        // 0.02). Crucially this OVERRIDES the Z-tier F3..F5 (e.g. 0.005 for a ~2-20 Mb
        // search space) — omitting it made --mid ~4x too strict on Forward-bias (F3b),
        // dropping genome-scale hits whose F3b P sits between 0.005 and 0.02.
        let fmid = cfg.fmid.unwrap_or(0.02);
        if do_mid {
            // F1/F2 are unused (MSV/Viterbi off), but mirror C's F1=F2=1.0 for clarity.
            f1 = 1.0; do_vit = false; do_vitbias = false; f2 = 1.0; f2b = 1.0;
            f3 = fmid; f3b = fmid; f4 = fmid; f4b = fmid; f5 = fmid;
        } else if cfg.rfam && !do_max && !nohmm {
            f1 = 0.06; do_vit = true; do_vitbias = true; f2 = 0.02; f2b = 0.02;
            f3 = 0.0002; f3b = 0.0002; f4 = 0.0002; f4b = 0.0002; f5 = 0.0002;
        } else if z_mb >= (20000.0 - smallx1) {
            f1 = 0.06; do_vit = true; do_vitbias = true; f2 = 0.02; f2b = 0.02;
            f3 = 0.0002; f3b = 0.0002; f4 = 0.0002; f4b = 0.0002; f5 = 0.0002;
        } else if z_mb >= (2000.0 - smallx1) {
            f1 = 0.15; do_vit = true; do_vitbias = true; f2 = 0.15; f2b = 0.15;
            f3 = 0.0002; f3b = 0.0002; f4 = 0.0002; f4b = 0.0002; f5 = 0.0002;
        } else if z_mb >= (200.0 - smallx1) {
            f1 = 0.15; do_vit = true; do_vitbias = true; f2 = 0.15; f2b = 0.15;
            f3 = 0.0008; f3b = 0.0008; f4 = 0.0008; f4b = 0.0008; f5 = 0.0008;
        } else if z_mb >= (20.0 - smallx1) {
            f1 = 0.35; do_vit = true; do_vitbias = true; f2 = 0.15; f2b = 0.15;
            f3 = 0.003; f3b = 0.003; f4 = 0.003; f4b = 0.003; f5 = 0.003;
        } else if z_mb >= (2.0 - smallx1) {
            // C 544-545: do_vit = do_vitbias = FALSE; F2 = F2b = 1.0 (irrelevant).
            f1 = 0.35; do_vit = false; do_vitbias = false; f2 = 1.0; f2b = 1.0;
            f3 = 0.005; f3b = 0.005; f4 = 0.005; f4b = 0.005; f5 = 0.005;
        } else {
            // C 553-554: do_vit = do_vitbias = FALSE; F2 = F2b = 1.0 (irrelevant).
            f1 = 0.35; do_vit = false; do_vitbias = false; f2 = 1.0; f2b = 1.0;
            f3 = 0.02; f3b = 0.02; f4 = 0.02; f4b = 0.02; f5 = 0.02;
        }
        // C cm_pipeline.c:443/457/472: --max/--nohmm/--mid turn OFF the Viterbi filter
        // (do_vit = do_vitbias = FALSE), overriding the tier/rfam defaults.
        if do_max || nohmm || do_mid {
            do_vit = false;
            do_vitbias = false;
        }
        // C expert Viterbi threshold overrides (cm_pipeline.c:577-578): --F2/--F2b turn
        // the stage ON and set its threshold, only when NOT --max/--nohmm/--mid.
        if !do_max && !nohmm && !do_mid {
            if let Some(v) = cfg.f2 { do_vit = true; f2 = v.min(1.0); }
            if let Some(v) = cfg.f2b { do_vitbias = true; f2b = v.min(1.0); }
        }
        // C cm_pipeline.c:593/601: --noF2/--noF2b turn the stage OFF (applied after the
        // --F2/--F2b enables, so --noF2 wins if both are somehow set).
        if cfg.no_f2 { do_vit = false; }
        if cfg.no_f2b { do_vitbias = false; }
        // C do_msvbias / F1b: every default tier sets do_msvbias=FALSE with F1b=1.0
        // (or 0.35 for the 20-200 Mb tier, cm_pipeline.c:507-508/537). --doF1b/--F1b
        // enable it (only when !max/!nohmm/!mid, cm_pipeline.c:576/599); --F1b also sets
        // the threshold, clamped to ESL_MIN(1.0, x).
        let mut do_msvbias = false;
        let mut f1b: f64 = if z_mb >= (20.0 - smallx1) && z_mb < (200.0 - smallx1) { 0.35 } else { 1.0 };
        if !do_max && !nohmm && !do_mid {
            if let Some(v) = cfg.f1b { do_msvbias = true; f1b = v.min(1.0); }
            if cfg.do_f1b { do_msvbias = true; }
        }
        // C do_edefbias / F5b (cm_pipeline.c:438/586/604): OFF by default. Every tier
        // sets `F5 = F5b` together (e.g. line 559 `pli->F5 = pli->F5b = 0.02`), so the
        // default F5b tracks the tier F5 (captured here BEFORE the --F5 override, which
        // only touches F5). --F5b sets it ON with its own threshold; --doF5b turns it ON
        // at the tier default. Both only when !max/!nohmm. --mid does NOT gate F5/F5b.
        let mut do_edefbias = false;
        let mut f5b: f64 = f5;
        if !do_max && !nohmm {
            if let Some(v) = cfg.f5b { do_edefbias = true; f5b = v.min(1.0); }
            if cfg.do_f5b { do_edefbias = true; }
        }
        // C do_fwd/do_fwdbias/do_gfwd/do_gfwdbias (cm_pipeline.c:433-436 base TRUE;
        // 443/457 --max/--nohmm force FALSE; 594-603 --noF3/--noF3b/--noF4/--noF4b force
        // FALSE). --F3/--F3b/--F4/--F4b re-enable (redundant here; the threshold is
        // handled by apply_filter_overrides). --mid does NOT affect these (it only
        // gates F1/F2). do_gfwd only drives display/counting (no filter gate in C).
        let hmm_filters = !do_max && !nohmm;
        let do_fwd = hmm_filters && !cfg.no_f3;
        let do_fwdbias = hmm_filters && !cfg.no_f3b;
        let do_gfwd = hmm_filters && !cfg.no_f4;
        let do_gfwdbias = hmm_filters && !cfg.no_f4b;
        let e_report = cfg.e_report;
        // --max/--nohmm/--mid remain `-g`-only FOR NOW. The local-begin scan DP
        // (cm_nohmm::generic_scan) + cm_local_scan infrastructure below is byte-verified
        // and kept in as inert scaffolding for the local nohmm/max final stage, but the
        // local path is NOT enabled until the faithful QDB-banded / EL-aware D&C aligner
        // (cm_dpsmall) makes local --max/--nohmm whole-file byte-identical to C. Until
        // then, local --max/--nohmm fall through to the default local pipeline exactly
        // as on main (no half-working feature shipped). See STEP 2.
        // --nohmm / --max are now faithful in BOTH glocal (-g) and the default local
        // config: local begins in the nonbanded/QDB scanner (cm_nohmm::generic_scan)
        // + EL-aware QDB/non-banded D&C alignment (cyk_align_maybe_banded) + faithful
        // ParsetreeToCMBounds display. Verified whole-file byte-identical vs C.
        // (nohmm/do_max/do_mid hoisted above the tier block.)
        // C per-stage filter P-value overrides (cm_pipeline.c:574-586), clamped to
        // ESL_MIN(1.0, x). F3..F5 apply when NOT --max/--nohmm; F1 also excludes --mid.
        let (f1, f3, f3b, f4, f4b, f5) =
            apply_filter_overrides(cfg, do_max, nohmm, do_mid, f1, f3, f3b, f4, f4b, f5);
        // `--nohmm` disables truncation in C (see search()); exclude it from do_gtrunc
        // so `-g --nohmm` uses the STD-only nohmm path. `--anytrunc/--inttrunc/
        // --onlytrunc` (cfg.notrunc==false) also route here in GLOBAL config.
        // C cm_pipeline.c:318-374: truncation config depends ONLY on the trunc options
        // (--anytrunc/--notrunc/…), NOT on --mid or -g. So `-g --mid` runs the DEFAULT
        // truncated pipeline (do_trunc_ends=TRUE) with do_msv=FALSE — the same glocal
        // STD+5P/3P/53 machinery as `-g`, not a separate reduced path. (`--mid` only
        // turns off MSV/Viterbi and sets the Fwd thresholds; verified by C stats showing
        // 4332 residues re-searched for truncated hits under `-g --mid`.)
        let do_gtrunc = cfg.global && !nohmm && !cfg.max && !cfg.notrunc;
        // Default LOCAL truncated search: not `-g`, truncation not disabled. C sets
        // pli->do_trunc_ends=TRUE and CM_CONFIG_LOCAL for a non-`-g` cmsearch
        // (cm_pipeline.c:367-374, 722-732), so the same STD+5P/3P/53 pass loop runs
        // in LOCAL config. C disables truncation under --max/--nohmm (do_trunc_ends
        // FALSE, cm_pipeline.c:441-455/580) → the STD-only nohmm/max scan path runs
        // and the footer reports "re-searched 0 residues".
        let do_ltrunc = !cfg.global && !cfg.notrunc && !nohmm && !do_max;
        // C `--notrunc` (do_trunc_ends=FALSE): run only the STD pass, but through the
        // SAME CM pipeline (proper alidisplay), in LOCAL (default) or GLOBAL (`-g`)
        // config. Excludes the -g-only --max/--nohmm/--mid modes (those have their own
        // paths). Covers both `--notrunc` and `-g --notrunc`.
        let do_notrunc_cm = cfg.notrunc && !nohmm && !do_max; // NOT !do_mid: -g --mid --notrunc must take the notrunc-CM STD path (C runs STD pass, MSV/Vit off), else it falls to the default pipeline and under-scores (clips terminal model positions)
        // C cm_pipeline.c:318-374 truncation-mode selection (same if/else precedence:
        // any > int > only > notrunc > 5trunc > 3trunc > default). --anytrunc/--inttrunc/
        // --onlytrunc add the internal PLI_PASS_5P_AND_3P_ANY pass, which uses the LOCAL
        // p7 domain-def engine (p7_domaindef_local, ported for --hmmonly) — see
        // pass5_env_def + the do_any handling in trunc_pipeline_one_strand.
        let trunc_mode = if cfg.anytrunc { TruncMode::AnyTrunc }
            else if cfg.inttrunc { TruncMode::IntTrunc }
            else if cfg.onlytrunc { TruncMode::OnlyTrunc }
            else if cfg.notrunc { TruncMode::Notrunc }
            else if cfg.trunc5p { TruncMode::Trunc5 }
            else if cfg.trunc3p { TruncMode::Trunc3 }
            else { TruncMode::Default };
        // C cm_pipeline.c:743-758: when --qdb (or --max/--nohmm) is set, the final
        // Inside round uses QDBs, hit alignment is forced to nonbanded CYK D&C, and
        // ALL truncation is DISABLED ("D&C truncated alignment is not robust, so we
        // don't allow it": do_trunc_ends/any/int/only/5p_ends/3p_ends all = FALSE).
        // Note --nonbanded does NOT trigger this block: it keeps default truncation on.
        let trunc_mode = if cfg.qdb || do_max || nohmm { TruncMode::Notrunc } else { trunc_mode };
        // The CM (and thus the final exp-tail) is glocal iff `-g`. This equals the old
        // `do_gtrunc || nohmm || do_max || do_mid` for every previously-routed case and
        // additionally gives `-g --notrunc` its correct GLOBAL exp (was a latent gap).
        let is_global_exp = cfg.global;
        // C cm_pipeline.c:1005/1013: final_cm_exp_mode = (opts & CM_SEARCH_INSIDE)
        // ? {GI|LI} : {GC|LC}. `--cyk` clears INSIDE (696), so the final round uses
        // the CYK exp-tail (GC glocal / LC local) instead of Inside (GI/LI).
        let exp_final: ExpParams = match (is_global_exp, cfg.cyk) {
            (true, false) => self.gi.clone(),
            (true, true) => self.gc.clone(),
            (false, false) => self.li.clone(),
            (false, true) => self.lc.clone(),
        };
        let ez_final: f64 = (z / exp_final.dbsize) * exp_final.nrandhits as f64;
        let pli_t: f32 = match self.resolved_t_cutoff(cfg) {
            Some(t) => t,
            None => (exp_final.mu + (e_report / ez_final).ln() / (-1.0 * exp_final.lambda)) as f32,
        };
        // ---- HMM-only (--hmmonly / 0-basepair model) config ----------------------
        // C cm_pli_NewModel do_hmmonly_cur gating (cm_pipeline.c:1028-1030): number of
        // consensus basepairs = number of MATP nodes.
        let cm_nbp = self
            .cm
            .ndtype
            .iter()
            .filter(|&&t| t as i32 == crate::constants::MATP_ND)
            .count() as i32;
        // C cm_pipeline.c:1028-1030: do_hmmonly_never = (--nohmmonly || --max || --nohmm);
        // do_hmmonly_always = --hmmonly ONLY (--hmmmax does NOT enable HMM-only; it only
        // sets the max filter params once HMM-only is already on). do_glocal_cm_cur (-g)
        // forces the CM path.
        let hmmonly = crate::p7_hmmonly::newmodel_do_hmmonly_cur(
            cfg.nohmmonly || cfg.max || cfg.nohmm,
            cfg.global,
            cfg.hmmonly,
            cm_nbp,
        );
        // C cm_pipeline.c:613-631 filter thresholds + null2/bias toggles.
        let hmm_fcfg = crate::p7_hmmonly::hmmonly_filter_cfg(
            cfg.hmm_f1.unwrap_or(0.02) as f32,
            cfg.hmm_f2.unwrap_or(1e-3) as f32,
            cfg.hmm_f3.unwrap_or(1e-5) as f32,
            cfg.hmmmax,
            cfg.hmmnonull2,
            cfg.hmmnobias,
        );
        // HMM-only E-value machinery (nhmmer convention): eZ = pli->Z / (float)max_length
        // (cmsearch.c:651); pli->T = cm_p7_E2Score(E, Z, max_length, LFTAU, LFLAMBDA)
        // (cm_pipeline.c:1069, stats.c:346); pvalue = esl_exp_surv(score, LFTAU, LFLAMBDA);
        // evalue = pvalue * eZ (cm_tophits.c:938). Overrides the CM exp-tail for this model.
        let p7_maxl = self.cm.p7.as_ref().map(|p| p.max_length).unwrap_or(1).max(1);
        // eZ for E-values: C does (int64)pli->Z / (float)max_length (single precision).
        let ez_hmm = ((z as f32) / (p7_maxl as f32)) as f64;
        // pli->T: cm_p7_E2Score uses double Z / (double)(float)hitlen.
        let pli_t_hmm: f32 = match self.resolved_t_cutoff(cfg) {
            Some(t) => t,
            None => (self.lftau
                + (e_report / (z / ((p7_maxl as f32) as f64))).ln() / (-1.0 * self.lflambda))
                as f32,
        };
        let (exp_mu, exp_lambda, ez_final, pli_t) = if hmmonly {
            (self.lftau, self.lflambda, ez_hmm, pli_t_hmm)
        } else {
            (exp_final.mu, exp_final.lambda, ez_final, pli_t)
        };
        // C `--F6` (cm_pipeline.c:589): applied when NOT `--max`, clamped ESL_MIN(1.0,x).
        // Tier default is 0.0001 in every non-max strategy (461/477/486/517..559).
        let f6: f64 = if !do_max {
            cfg.f6.map(|x| x.min(1.0)).unwrap_or(0.0001)
        } else {
            1.0 // do_max: all filters off; F6 unused (cm_pipeline.c:441-446)
        };
        // C `--cykenvx <n>` default 10 (cm_pipeline.c:648): F6env = ESL_MIN(1.0, F6*n).
        // NOTE: C casts n to (float) then multiplies; keep the f32 product then min.
        let cykenvx = cfg.cykenvx.unwrap_or(10);
        let f6env: f64 = ((f6 as f32) * (cykenvx as f32)).min(1.0) as f64;
        // CYK envelope-redefinition cutoffs (cm_pipeline.c:3405): mu + log(F6env)/-lambda,
        // per exp mode: EXP_CM_LC (local std) and EXP_CM_GC (glocal / nohmm / mid).
        let cyk_env_cutoff: f32 = (self.lc.mu + f6env.ln() / (-1.0 * self.lc.lambda)) as f32;
        let gcyk_env_cutoff: f32 = (self.gc.mu + f6env.ln() / (-1.0 * self.gc.lambda)) as f32;
        // nohmm CYK filter cutoff (cm_pipeline.c:3557): mu + log(F6)/-lambda (uses F6,
        // not F6env). fcyk_cm_exp_mode is EXP_CM_GC in glocal (`-g`/mid) config and
        // EXP_CM_LC in the default LOCAL config, so pick the local CYK exp-tail (LC)
        // for a non-`-g` --nohmm run and the global (GC) for `-g`.
        let cyk_filt_exp = if cfg.global { &self.gc } else { &self.lc };
        let cyk_cutoff: f32 = (cyk_filt_exp.mu + f6.ln() / (-1.0 * cyk_filt_exp.lambda)) as f32;
        // C `--noF6` (596): do_fcyk=FALSE; `--nocykenv` (646): do_fcykenv=FALSE.
        // C `--max` (cm_pipeline.c:443) turns ALL filters off, including the CYK filter
        // (do_fcyk=FALSE), so the CYK-filter stats line renders "(off)".
        let do_fcyk = !cfg.no_f6 && !do_max;
        let do_fcykenv = !cfg.nocykenv && !do_max;
        // HMM-band tau overrides (cm_pipeline.c:645/651/240); defaults match the
        // searcher's model-derived constants.
        let fcyk_tau = cfg.ftau.unwrap_or(self.fcyk_tau);
        let final_tau = cfg.tau.unwrap_or(self.final_tau);
        let maxtau = cfg.maxtau.unwrap_or(self.maxtau);
        // Glocal env-def region thresholds (cm_pipeline.c:309-312); defaults 0.25/0.10/0.20/200.
        let rt1 = cfg.rt1.map(|x| x as f32).unwrap_or(0.25);
        let rt2 = cfg.rt2.map(|x| x as f32).unwrap_or(0.10);
        let rt3 = cfg.rt3.map(|x| x as f32).unwrap_or(0.20);
        let ns = cfg.ns.map(|x| x as usize).unwrap_or(200);
        Thresh {
            f1, do_msvbias, f1b, do_vit, f2, do_vitbias, f2b, do_fwd, do_fwdbias, do_gfwd, do_gfwdbias,
            do_edefbias, f5b,
            f3, f3b, f4, f4b, f5, pli_t, cyk_cutoff,
            f6, cyk_env_cutoff, gcyk_env_cutoff, do_fcyk, do_fcykenv,
            fcyk_tau, final_tau, maxtau,
            rt1, rt2, rt3, ns,
            do_gtrunc, do_ltrunc, do_notrunc_cm, trunc_mode, global: cfg.global, do_max, nohmm, do_mid,
            // C pli->do_msv = std_filters(!max && !nohmm) && !mid.
            do_msv: !do_max && !nohmm && !cfg.mid && !cfg.no_f1,
            do_cyk: cfg.cyk,
            // C cm_pipeline.c:705-708: default-config final round is QDB/nonbanded only
            // via --qdb/--nonbanded (not under --max/--nohmm, which own their paths).
            do_qdb: cfg.qdb && !do_max && !nohmm,
            do_nonbanded: cfg.nonbanded && !do_max && !nohmm,
            e_report, t_cutoff: self.resolved_t_cutoff(cfg),
            exp_mu, exp_lambda, ez_final,
            hmmonly,
            hmm_f1: hmm_fcfg.f1 as f64,
            hmm_f2: hmm_fcfg.f2 as f64,
            hmm_f3: hmm_fcfg.f3 as f64,
            hmm_do_max: hmm_fcfg.do_max,
            hmm_do_null2: hmm_fcfg.do_null2,
            // C cm_pipeline.c:640: pli->do_null3 = --nonull3 ? FALSE : TRUE.
            do_null3: !cfg.nonull3,
        }
    }

    /// Build the independent (chunk × strand) window task list for `fulls`,
    /// exactly as `search` does (C ReadWindow: CM_MAX_RESIDUE_COUNT residues,
    /// maxw overlap). Each task is a self-contained window.
    fn window_tasks(
        &self,
        fulls: &[Vec<u8>],
        toponly: bool,
        bottomonly: bool,
    ) -> Vec<(usize, usize, usize, usize, bool, usize)> {
        // C do_top = !bottomonly, do_bot = !toponly (cm_pipeline.c:235-236).
        let do_top = !bottomonly;
        let do_bot = !toponly;
        // `ctx` (the maxW overlap context shared with the previous chunk, C `dbsq->C`)
        // is threaded into the task so the STD-pass residue accounting can subtract it
        // (C cm_pli_AdjustNresForOverlaps): overlaps are never counted twice.
        let mut tasks: Vec<(usize, usize, usize, usize, bool, usize)> = Vec::new();
        for (seq_idx, full) in fulls.iter().enumerate() {
            let l = full.len() - 2;
            let mut new_start = 1usize;
            let mut first = true;
            while new_start <= l {
                let ctx = if first { 0 } else { self.maxw.min(new_start - 1) };
                let wgs = new_start - ctx;
                let new_count = CM_MAX_RESIDUE_COUNT.min(l - new_start + 1);
                let win_len = ctx + new_count;
                let wge = wgs + win_len - 1;
                if do_top {
                    tasks.push((seq_idx, wgs, wge, win_len, false, ctx));
                }
                if do_bot {
                    tasks.push((seq_idx, wgs, wge, win_len, true, ctx));
                }
                new_start = wge + 1;
                first = false;
            }
        }
        tasks
    }

    /// Run the pipeline on a single (window × strand) task. This is the exact
    /// per-task body of `search`'s inner closure, extracted so both `search` and
    /// `search_many` share one copy → their results are byte-identical.
    fn run_window_task(
        &self,
        fulls: &[Vec<u8>],
        task: (usize, usize, usize, usize, bool, usize),
        th: &Thresh,
    ) -> Vec<Hit> {
        let (seq_idx, wgs, wge, win_len, in_rc, ctx) = task;
        let full = &fulls[seq_idx];
        let n = win_len + 2;
        let mut dsq = vec![255u8; n];
        if in_rc {
            for k in 1..n - 1 {
                let d = full[wge - (k - 1)];
                dsq[k] = if d < 4 { 3 - d } else { d };
            }
        } else {
            dsq[1..n - 1].copy_from_slice(&full[wgs..=wge]);
        }
        let hits = if th.hmmonly {
            // HMM-only pipeline (C pli_final_stage_hmmonly): the CM is never used.
            // Charged as the HMM_ONLY_ANY pass (nres = new chunk residues).
            self.acct.add_pass_res(
                crate::cm_trunc::PLI_PASS_HMM_ONLY_ANY as usize, in_rc, (win_len - ctx) as u64,
            );
            self.hmmonly_one_strand(&dsq, win_len, th)
        } else if th.do_gtrunc || th.do_ltrunc || th.do_notrunc_cm {
            let seq_l = (full.len() - 2) as usize;
            let (have5term, have3term) = if in_rc {
                (wge == seq_l, wgs == 1)
            } else {
                (wgs == 1, wge == seq_l)
            };
            // GLOBAL (`-g`) uses the GC/GI exp modes + global CM/CP9 + g_ptyAA penalties;
            // LOCAL (default) uses LC/LI + local CM/CP9(+EL) + l_ptyAA. The CYK-filter
            // envelope cutoff must match the mode (gcyk_env_cutoff is GC-based,
            // cyk_env_cutoff is LC-based); see cm_pipeline.c:770-775, 3392.
            // `--notrunc` (do_notrunc_cm) runs the same pipeline with do_trunc_ends=FALSE
            // (STD pass only) — C cm_pipeline.c:1375-1412 skips the 5P/3P/53 passes.
            let local = !th.global;
            let cyk_env_cutoff = if local { th.cyk_env_cutoff } else { th.gcyk_env_cutoff };
            self.trunc_pipeline_one_strand(
                &dsq, win_len, ctx, have5term, have3term, in_rc, local, th.f1, th.do_msvbias, th.f1b, th.do_vit, th.f2, th.do_vitbias, th.f2b, th.do_fwd, th.do_fwdbias, th.do_gfwdbias, th.do_edefbias, th.f5b, th.f3, th.f3b, th.f4, th.f4b, th.f5, th.pli_t,
                th.f6, cyk_env_cutoff, th.do_fcyk, th.do_fcykenv, th.fcyk_tau, th.final_tau, th.maxtau,
                th.rt1, th.rt2, th.rt3, th.ns, th.do_cyk, th.trunc_mode, th.do_qdb, th.do_nonbanded, th.do_msv, th.do_null3,
            )
        } else if th.do_max {
            // C cm_pipeline.c:1444-1446 — the STD pass (PLI_PASS_STD_ANY) charges nres =
            // sq->n; the maxW overlap context is subtracted (AdjustNresForOverlaps), so
            // the net charge is (win_len - ctx). The p7 filter accounting is skipped
            // (filters off). --max has no CYK filter (do_fcyk FALSE) so n_past_cyk is
            // left unset → the summary shows "(off)".
            self.acct.add_pass_res(crate::cm_trunc::PLI_PASS_STD_ANY as usize, in_rc, (win_len - ctx) as u64);
            self.max_one_window(&dsq, win_len, th.pli_t, th.global, th.do_null3)
        } else if th.nohmm {
            self.acct.add_pass_res(crate::cm_trunc::PLI_PASS_STD_ANY as usize, in_rc, (win_len - ctx) as u64);
            self.nohmm_one_window(&dsq, win_len, th.cyk_cutoff, th.pli_t, th.do_fcyk, th.global, th.do_null3)
        } else {
            let mut gm = self.gm_proto.clone();
            self.pipeline_one_strand(
                &mut gm, &dsq, win_len, th.f1, th.do_msvbias, th.f1b, th.do_vit, th.f2, th.do_vitbias, th.f2b, th.do_fwd, th.do_fwdbias, th.f3, th.f3b, th.f4, th.f4b, th.f5, th.pli_t,
                th.f6, th.cyk_env_cutoff, th.do_fcyk, th.do_fcykenv, th.fcyk_tau, th.final_tau, th.maxtau,
                th.rt1, th.rt2, th.rt3, th.ns, th.do_msv, th.do_null3,
            )
        };
        hits.into_iter()
            .map(|(ws_loc, we_loc, sc, bias, mdl_from, mdl_to, ad)| {
                let (gstart, gstop) = if in_rc {
                    (wge as i64 - ws_loc as i64 + 1, wge as i64 - we_loc as i64 + 1)
                } else {
                    (wgs as i64 + ws_loc as i64 - 1, wgs as i64 + we_loc as i64 - 1)
                };
                let pvalue = esl_exp_surv(sc as f64, th.exp_mu, th.exp_lambda);
                let evalue = pvalue * th.ez_final;
                let gc = hit_gc(full, gstart, gstop);
                let (trunc, pass_idx) = ad
                    .as_ref()
                    .map(|a| (a.trunc.clone(), a.pass_idx))
                    .unwrap_or_else(|| ("no".to_string(), 1));
                Hit {
                    seq_idx,
                    in_rc,
                    start: gstart,
                    stop: gstop,
                    score: sc,
                    bias,
                    mdl_from,
                    mdl_to,
                    gc,
                    pvalue,
                    evalue,
                    trunc,
                    pass_idx,
                    hmmonly: th.hmmonly,
                    removed: false,
                    alignment: ad,
                }
            })
            .collect::<Vec<_>>()
    }

    /// Global overlap removal + SortByEvalue + reporting threshold, exactly as the
    /// tail of `search`. `all_hits` are all the hits for a SINGLE model.
    fn finalize(&self, mut all_hits: Vec<Hit>, th: &Thresh) -> Vec<FaithfulHit> {
        remove_overlaps_global(&mut all_hits);
        all_hits.retain(|h| !h.removed);
        // C hit_sorter_by_evalue (cm_tophits.c:193-217), the exact 5-level chain:
        //   1) evalue ascending  2) score descending  3) seq_idx ascending
        //   4) start ascending    5) pass_idx descending  (then qsort-arbitrary).
        // We append (stop asc) only as a final total-order tiebreak for cross-thread
        // determinism; it never fires for cases C resolves within the 5 keys.
        all_hits.sort_by(|a, b| {
            a.evalue
                .partial_cmp(&b.evalue)
                .unwrap()
                .then(b.score.partial_cmp(&a.score).unwrap())
                .then(a.seq_idx.cmp(&b.seq_idx))
                .then(a.start.cmp(&b.start))
                .then(b.pass_idx.cmp(&a.pass_idx))
                .then(a.stop.cmp(&b.stop))
        });
        all_hits
            .into_iter()
            .filter(|h| match th.t_cutoff {
                Some(t) => h.score >= t,
                None => h.evalue <= th.e_report,
            })
            .map(|h| FaithfulHit {
                seq_idx: h.seq_idx,
                in_rc: h.in_rc,
                start: h.start,
                stop: h.stop,
                score: h.score,
                bias: h.bias,
                mdl_from: h.mdl_from,
                mdl_to: h.mdl_to,
                gc: h.gc,
                pvalue: h.pvalue,
                evalue: h.evalue,
                trunc: h.trunc,
                pass_idx: h.pass_idx,
                hmmonly: h.hmmonly,
                alignment: h.alignment,
            })
            .collect()
    }

    /// Build the STD-path Viterbi (F2) / Viterbi-bias (F2b) filter parameters for
    /// [`f3_filter_sequence`]. `None` when `do_vit` is off (the pre-existing behavior
    /// for Z_Mb < 20 and --max/--nohmm/--mid), so those callers stay byte-unchanged.
    fn std_vit_params(
        &self,
        do_vit: bool,
        f2: f64,
        do_vitbias: bool,
        f2b: f64,
    ) -> Option<crate::cm_pipeline::VitParams<'_>> {
        if do_vit {
            Some(crate::cm_pipeline::VitParams {
                vf: &self.vf,
                f2,
                do_vitbias,
                f2b,
                lvmu: self.lvmu,
                lvlambda: self.lvlambda,
            })
        } else {
            None
        }
    }

    /// Build the F1b (MSV composition-bias) sub-filter parameters for
    /// [`f3_filter_sequence`]. `None` when `do_msvbias` is off (the default), so those
    /// callers stay byte-unchanged.
    fn std_msvbias_params(&self, do_msvbias: bool, f1b: f64) -> Option<crate::cm_pipeline::MsvBiasParams> {
        if do_msvbias {
            Some(crate::cm_pipeline::MsvBiasParams {
                f1b,
                lmmu: self.lmmu,
                lmlambda: self.lmlambda,
            })
        } else {
            None
        }
    }

    /// Run LOOP-1 (F1/F3/F3b + F4/F4b/F5) then LOOP-2 (F6 + F7) on one strand of
    /// one window. Returns final-stage hits in window-local coords:
    /// (start, stop, score, bias, mdl_from, mdl_to).
    #[allow(clippy::too_many_arguments)]
    fn pipeline_one_strand(
        &self,
        gm: &mut GlocalProfile,
        wdsq: &[u8],
        win_len: usize,
        f1: f64,
        do_msvbias: bool,
        f1b: f64,
        do_vit: bool,
        f2: f64,
        do_vitbias: bool,
        f2b: f64,
        do_fwd: bool,
        do_fwdbias: bool,
        f3: f64,
        f3b: f64,
        f4: f64,
        f4b: f64,
        f5: f64,
        pli_t: f32,
        f6: f64,
        cyk_env_cutoff: f32,
        do_fcyk: bool,
        do_fcykenv: bool,
        fcyk_tau: f64,
        final_tau: f64,
        maxtau: f64,
        rt1: f32,
        rt2: f32,
        rt3: f32,
        ns: usize,
        do_msv: bool,
        do_null3: bool,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        let cm = &self.cm;
        let cp9 = &self.cp9;
        let map = &self.map;
        let size_limit = self.size_limit;
        let gfmu = self.gfmu;
        let gflambda = self.gflambda;
        let ln2 = self.ln2;
        let lc = &self.lc;

        // ---- LOOP-1: F1/F3/F3b merged windows ----
        let _t = std::time::Instant::now();
        let (merged, _f1f3) = f3_filter_sequence(&self.mf, &self.ff, &self.bf, wdsq, win_len, f1, f3, f3b, cm.w as i64,
            self.std_msvbias_params(do_msvbias, f1b), self.std_vit_params(do_vit, f2, do_vitbias, f2b), do_fwd, do_fwdbias, do_msv, self.maxw);
        if std::env::var("IX_DEBUG").is_ok() {
            let tot: i64 = merged.iter().map(|w| (w.end - w.start + 1) as i64).sum();
            eprintln!("[F1F3] win_len={} gm.m={} merged_wins={} total_merged_len={} t={:.2}s",
                win_len, gm.m, merged.len(), tot, _t.elapsed().as_secs_f64());
        }

        // ---- F4/F4b/F5: envelope definition per surviving window ----
        let mut p7envs: Vec<(i32, i32)> = Vec::new(); // (es, ee) window-local
        for w in merged.iter() {
            let ws = w.start;
            let we = w.end;
            let wlen = (we - ws + 1) as usize;

            let mut sub = vec![255u8];
            sub.extend_from_slice(&wdsq[ws as usize..=we as usize]);
            sub.push(255u8);

            if std::env::var("IX_DEBUG").is_ok() {
                eprintln!("[F4F5] window ws={} we={} wlen={} (gm.m={}) -> gforward...", ws, we, wlen, gm.m);
            }
            let nullsc = p7_bg_null_one(wlen);
            reconfig_length(gm, wlen as i32);
            let mut gx = P7Gmx::new(gm.m, wlen);
            let _tw = std::time::Instant::now();
            let fwdsc = p7_gforward(&sub, wlen, gm, &mut gx);
            if std::env::var("IX_DEBUG").is_ok() {
                eprintln!("[F4F5]   gforward done t={:.2}s", _tw.elapsed().as_secs_f64());
            }

            // F4: glocal Forward P-value
            let sc = (fwdsc as f64 - nullsc) / ln2;
            if esl_exp_surv(sc, gfmu, gflambda) > f4 {
                continue;
            }

            // F4b: glocal-Forward composition bias
            let filtersc = bias_filter_score(&self.bf, &sub, wlen);
            let sc_b = (fwdsc as f64 - filtersc as f64) / ln2;
            if esl_exp_surv(sc_b, gfmu, gflambda) > f4b {
                continue;
            }

            // Backward (fills matrix for domain def)
            let mut gxb = P7Gmx::new(gm.m, wlen);
            let _bcksc = p7_gbackward(&sub, wlen, gm, &mut gxb);

            // glocal domain definition (do_null2 off by default)
            let do_null2 = false;
            // STD/mid glocal profile is multihit → is_unihit=false (reconfig to unihit).
            let domains = p7_domaindef_glocal(gm, &sub, wlen, &gx, &gxb, do_null2, rt1, rt2, rt3, ns, false);
            let omega = 1.0f32 / 256.0f32;
            for dom in domains.iter() {
                let env_len = dom.jenv - dom.ienv + 1;
                let env_sc = dom.envsc
                    + (wlen as f32 - env_len as f32) * ((wlen as f32) / (wlen as f32 + 3.0)).ln();
                let env_edefbias = if do_null2 {
                    p7_flogsum(0.0, omega.ln() + dom.domcorrection)
                } else {
                    0.0
                };
                let sc5 = (env_sc as f64 - (nullsc + env_edefbias as f64)) / ln2;
                if esl_exp_surv(sc5, gfmu, gflambda) <= f5 {
                    let es = (dom.ienv as i64 + ws - 1) as i32;
                    let ee = (dom.jenv as i64 + ws - 1) as i32;
                    p7envs.push((es, ee));
                }
            }
        }
        if p7envs.is_empty() {
            return Vec::new();
        }

        // ---- LOOP-2 / F6: CYK env filter ----
        // C `--noF6` (do_fcyk=FALSE): skip the CYK filter entirely — the F5-surviving
        // envelopes pass straight to the final stage (cm_pipeline.c: pli_cyk_env_filter
        // is only called when pli->do_fcyk).
        let mut surv_env: Vec<(i32, i32)> = Vec::new();
        if !do_fcyk {
            surv_env = p7envs.clone();
        } else {
        if std::env::var("IX_DEBUG").is_ok() {
            eprintln!("[F6] n_p7envs={} size_limit={:.0}MB", p7envs.len(), size_limit);
        }
        for &(mut es, mut ee) in p7envs.iter() {
            let _tb = std::time::Instant::now();
            let (cp9b, _tau, mb) =
                cp9_iterate_seq2bands(cm, cp9, map, wdsq, es, ee, fcyk_tau, maxtau, size_limit, true, true);
            if std::env::var("IX_DEBUG").is_ok() {
                eprintln!("[F6] env es={} ee={} len={} band_mb={:.1} band_t={:.2}s",
                    es, ee, ee - es + 1, mb, _tb.elapsed().as_secs_f64());
            }
            if mb > size_limit {
                continue;
            } // eslERANGE overflow: skip envelope
            let _tc = std::time::Instant::now();
            let (sc, envi, envj) = fast_cyk_scan_hb(cm, &cp9b, wdsq, es, ee, cyk_env_cutoff);
            if std::env::var("IX_DEBUG").is_ok() {
                eprintln!("[F6] CYK es={} ee={} cyk_t={:.2}s sc={:.1}", es, ee, _tc.elapsed().as_secs_f64(), sc);
            }
            let p = esl_exp_surv(sc as f64, lc.mu, lc.lambda);
            if p > f6 {
                continue;
            }
            // do_fcykenv ON: refine envelope boundaries (C `--nocykenv` leaves es/ee).
            if do_fcykenv && envi != -1 && envj != -1 {
                es = envi as i32;
                ee = envj as i32;
            }
            surv_env.push((es, ee));
        }
        }
        if surv_env.is_empty() {
            return Vec::new();
        }

        // ---- F7: final Inside stage ----
        // emit map is CM-only; build once and reuse for per-hit ParsetreeToCMBounds.
        let emap = crate::create_emit_map(cm).expect("create_emit_map");
        let mut out: Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> =
            Vec::new();
        for &(es, ee) in surv_env.iter() {
            let (cp9b, _tau, mb) =
                cp9_iterate_seq2bands(cm, cp9, map, wdsq, es, ee, final_tau, maxtau, size_limit, true, true);
            if mb > size_limit {
                continue;
            }
            let (_sc, _ei, _ej, raw) =
                fast_finside_scan_hb(cm, &cp9b, wdsq, es, ee, 0.0, pli_t, do_null3);
            let surv = remove_overlaps_greedy(raw);
            // Per surviving hit: HMM-banded CYK align (shifted bands) -> mdl from/to.
            // C: pli_align_hit -> cp9_ShiftCMBands -> DispatchSqAlignment -> ParsetreeToCMBounds.
            for (hi, hj, hsc, hbias) in surv {
                let mut cb = cp9b.clone();
                crate::cp9::shift_cm_bands(cm, &mut cb, hi, hj);
                let lp = hj - hi + 1;
                let (cfrom, cto) = crate::cp9::cyk_align_hb_cmbounds(
                    cm,
                    &cb,
                    &wdsq[(hi as usize - 1)..],
                    lp,
                    &emap,
                );
                out.push((hi, hj, hsc, hbias, cfrom, cto, None));
            }
        }
        out
    }

    /// Faithful `cm_Pipeline` (cm_pipeline.c:1284) for the default truncation
    /// (`do_trunc_ends`): the STANDARD pass plus the three truncated passes
    /// (5P_ONLY_FORCE / 3P_ONLY_FORCE / 5P_AND_3P_FORCE). `local` selects the CM
    /// configuration: `false` = GLOBAL (`-g`, GC/GI exp, global CM/CP9, g_ptyAA);
    /// `true` = LOCAL (default cmsearch, LC/LI exp, local CM/CP9+EL, l_ptyAA). The
    /// pass-selection, accounting, and HMM env-def stages are config-independent
    /// (the p7 filter is always local); only the CM DP stages differ by `local`.
    /// `wdsq` is one chunk (C's `sq`); `have5term`/`have3term` say whether it touches
    /// the parent sequence's 5'/3' ends (cm_pipeline.c:1338-1345). Returns final-stage
    /// hits in chunk-local coords: (start, stop, score, bias, mdl_from, mdl_to, ad),
    /// with `ad` carrying trunc + pass_idx (None ⇒ trunc="no", pass=1).
    ///
    /// Structure mirrors C: two conceptual loops (HMM stages, then CM stages), fused
    /// per-pass here because infernox processes one strand/chunk at a time. The p7
    /// filter (F1/F3) is shared across passes when sq2search is the whole chunk (short
    /// seqs, `sq->n <= maxW`).
    #[allow(clippy::too_many_arguments)]
    fn trunc_pipeline_one_strand(
        &self,
        wdsq: &[u8],
        win_len: usize,
        ctx: usize,
        have5term: bool,
        have3term: bool,
        in_rc: bool,
        local: bool,
        f1: f64,
        do_msvbias: bool,
        f1b: f64,
        do_vit: bool,
        f2: f64,
        do_vitbias: bool,
        f2b: f64,
        do_fwd: bool,
        do_fwdbias: bool,
        do_gfwdbias: bool,
        do_edefbias: bool,
        f5b: f64,
        f3: f64,
        f3b: f64,
        f4: f64,
        f4b: f64,
        f5: f64,
        pli_t: f32,
        f6: f64,
        gcyk_env_cutoff: f32,
        do_fcyk: bool,
        do_fcykenv: bool,
        fcyk_tau: f64,
        final_tau: f64,
        maxtau: f64,
        rt1: f32,
        rt2: f32,
        rt3: f32,
        ns: usize,
        do_cyk: bool,
        trunc_mode: TruncMode,
        do_qdb: bool,
        do_nonbanded: bool,
        do_msv: bool,
        do_null3: bool,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        use crate::cm_trunc::{
            PLI_PASS_STD_ANY, PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE,
            PLI_PASS_5P_AND_3P_FORCE, PLI_PASS_5P_AND_3P_ANY,
        };
        let maxw = self.maxw as i32;
        let l = win_len as i64;

        // Pass selection per C cm_pipeline.c:1371-1412 (mode-dependent). The FORCE
        // terminal passes need have5term/have3term (and 53 also n<=maxW). The internal
        // PLI_PASS_5P_AND_3P_ANY pass (do_any, do_local_envdef=TRUE) re-searches the
        // FULL sequence allowing truncation anywhere (enforce first/final = FALSE); it
        // reuses the byte-verified truncated CM scanner (TrCYKScanHB/FTrInsideScanHB,
        // Tcp9 bands, TRPENALTY_5P_AND_3P — identical fill/penalty as 53_FORCE, only the
        // enforce differs) driven by a LOCAL p7 domain-def (pass5_env_def).
        let (do_std, do_5p, do_3p, do_53, do_any) = match trunc_mode {
            // 1409-1411: STD only.
            TruncMode::Notrunc =>
                (true, false, false, false, false),
            // 1375-1381: default — STD + terminal 5P/3P/53 force.
            TruncMode::Default =>
                (true, have5term, have3term, have5term && have3term && l <= maxw as i64, false),
            // 1382-1385: STD + 5P force only.
            TruncMode::Trunc5 =>
                (true, have5term, false, false, false),
            // 1387-1390: STD + 3P force only.
            TruncMode::Trunc3 =>
                (true, false, have3term, false, false),
            // 1392-1399: --anytrunc — STD + terminal 5P/3P/53 force + internal ANY.
            TruncMode::AnyTrunc =>
                (true, have5term, have3term, have5term && have3term && l <= maxw as i64, true),
            // 1400-1404: --inttrunc — STD + internal ANY (no terminal FORCE).
            TruncMode::IntTrunc =>
                (true, false, false, false, true),
            // 1405-1408: --onlytrunc — internal ANY only (STD off).
            TruncMode::OnlyTrunc =>
                (false, false, false, false, true),
        };

        // C 1443-1452: npli/nres accounting per pass, per chunk, per strand — counted
        // for EVERY pass that runs, BEFORE any early-out (the C accounting sits above
        // the `nwin_pass_std_any==0 continue`). STD (a "..._ANY" pass) is charged the
        // full chunk (`sq->n`), but the maxW overlap context shared with the previous
        // chunk is then subtracted (C cm_pli_AdjustNresForOverlaps, cmsearch.c:834/849:
        // `nres_top/bot -= dbsq->C`) so overlaps aren't double-counted — the net STD
        // charge is the chunk's NEW residues (`win_len - ctx`). The FORCE truncated
        // passes are charged `min(maxW, n)` and get NO overlap adjustment (they never
        // search overlaps; C only adjusts STD_ANY / HMM_ONLY / 5P_AND_3P_ANY).
        let nres_std = (win_len - ctx) as u64;
        let nres_force = (self.maxw.min(win_len)) as u64;
        // STD and 5P_AND_3P_ANY are "..._ANY" passes charged nres_std (with overlap
        // adjustment); the FORCE passes are charged nres_force. Each is charged only
        // when it runs (C accounts inside the per-pass loop, after the `continue`s).
        if do_std { self.acct.add_pass_res(PLI_PASS_STD_ANY as usize, in_rc, nres_std); }
        if do_5p { self.acct.add_pass_res(PLI_PASS_5P_ONLY_FORCE as usize, in_rc, nres_force); }
        if do_3p { self.acct.add_pass_res(PLI_PASS_3P_ONLY_FORCE as usize, in_rc, nres_force); }
        if do_53 { self.acct.add_pass_res(PLI_PASS_5P_AND_3P_FORCE as usize, in_rc, nres_force); }
        // C 1444: PLI_PASS_5P_AND_3P_ANY is a "..._ANY" pass — charged the full chunk
        // (`sq->n`) with the same maxW overlap adjustment as STD (baked into nres_std;
        // cm_pli_AdjustNresForOverlaps subtracts noverlap from pass 5, cm_pipeline.c).
        if do_any { self.acct.add_pass_res(PLI_PASS_5P_AND_3P_ANY as usize, in_rc, nres_std); }

        // C 1547: pli_p7_filter on sq2search. For short seqs (l <= maxW) every pass's
        // sq2search is the whole chunk, so the F1/F3 windows are identical → compute
        // once. (Genome-scale 5P/3P term5/term3 subseqs are handled below.)
        let (whole_merged, whole_f1f3) =
            f3_filter_sequence(&self.mf, &self.ff, &self.bf, wdsq, win_len, f1, f3, f3b, self.cm.w as i64,
                self.std_msvbias_params(do_msvbias, f1b), self.std_vit_params(do_vit, f2, do_vitbias, f2b), do_fwd, do_fwdbias, do_msv, self.maxw);
        // C accounts the STD pass p7-filter survivors (n_past_msv += nwin, plus
        // n_past_fwd/fwdbias + pos_*) INSIDE pli_p7_filter (cm_pipeline.c:2638/2765/
        // 2785), i.e. for EVERY chunk that runs the filter — regardless of whether any
        // window ultimately survives Forward. So this tally must happen BEFORE the
        // "no F3b survivors" early return below, else chunks whose MSV windows all fail
        // Forward silently drop their n_past_msv (they're the vast majority at genome
        // scale). Charged only when the STD (whole-chunk ANY) pass runs.
        if do_std { self.acct.add_f1f3(PLI_PASS_STD_ANY as usize, &whole_f1f3); }
        // NOTE: pass 5's p7-filter msv/fwd charge is NOT applied here. C skips pass 5
        // entirely when the STD pass yields no F3b-surviving windows (nwin_pass_std_any
        // == 0, cm_pipeline.c:1454 `continue` — which sits BEFORE pass 5's pli_p7_filter
        // call), so pass 5's filter counters are only charged when whole_merged is
        // non-empty. That charge is applied in the `do_any` execution block below,
        // after the early return.
        // C 1454/1548: when the STD pass runs, nwin_pass_std_any = its F3b survivors;
        // if that's 0, the terminal FORCE passes AND pass 5 are skipped (their
        // pli_p7_filter is never reached). BUT --onlytrunc doesn't run STD, so
        // nwin_pass_std_any stays -1 (never gates) — pass 5 still runs its own filter
        // on the full sequence and charges its msv even with no survivors. So only
        // short-circuit here when the STD pass gated the truncated passes (do_std) and
        // found nothing; --onlytrunc (do_any && !do_std) always falls through to pass 5.
        if whole_merged.is_empty() && !(do_any && !do_std) {
            return Vec::new();
        }

        let mut out: Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> =
            Vec::new();

        // ---- STD pass (PLI_PASS_STD_ANY): glocal p7 env-def + CM stages ----
        if do_std && !whole_merged.is_empty() {
            // (n_past_msv/fwd/fwdbias already tallied above, matching C's ordering.)
            let envs = self.trunc_env_def(
                wdsq, win_len, &whole_merged, PLI_PASS_STD_ANY, f4, f4b, do_gfwdbias, f5, do_edefbias, f5b, rt1, rt2, rt3, ns,
            );
            self.cm_std_stage(
                wdsq, win_len, &envs, local, pli_t, f6, gcyk_env_cutoff, do_fcyk, do_fcykenv,
                fcyk_tau, final_tau, maxtau, do_cyk, do_qdb, do_nonbanded, do_null3, &mut out,
            );
        }

        // ---- Truncated passes: term5/term3 extraction + truncated env-def + stages ----
        // For short seqs sq2search is the whole chunk (start_offset=0). For long chunks
        // (l > maxW) 5P researches the 5'-most maxW residues, 3P the 3'-most maxW
        // residues (C 1462-1472); 53 is skipped (do_53 false when l > maxW).
        for &pass in &[PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE, PLI_PASS_5P_AND_3P_FORCE] {
            let run = match pass {
                x if x == PLI_PASS_5P_ONLY_FORCE => do_5p,
                x if x == PLI_PASS_3P_ONLY_FORCE => do_3p,
                _ => do_53,
            };
            if !run {
                continue;
            }
            // Determine sq2search + start_offset (C 1459-1472). subdsq is a
            // sentinel-padded window frame over the chosen residues.
            let (sub, sublen, start_offset): (Vec<u8>, usize, i64) =
                if l <= maxw as i64 || pass == PLI_PASS_5P_AND_3P_FORCE {
                    (wdsq.to_vec(), win_len, 0)
                } else if pass == PLI_PASS_5P_ONLY_FORCE {
                    // first maxW residues
                    let n = maxw as usize;
                    let mut s = vec![255u8; n + 2];
                    s[1..=n].copy_from_slice(&wdsq[1..=n]);
                    (s, n, 0)
                } else {
                    // 3P: final maxW residues; start_offset = win_len - maxW
                    let n = maxw as usize;
                    let off = win_len as i64 - maxw as i64;
                    let mut s = vec![255u8; n + 2];
                    s[1..=n].copy_from_slice(&wdsq[(off as usize + 1)..=(win_len)]);
                    (s, n, off)
                };
            // p7 filter on sq2search (identical to whole_merged when sub==chunk).
            let (merged, pass_f1f3) = if start_offset == 0 && sublen == win_len {
                (whole_merged.clone(), whole_f1f3)
            } else {
                f3_filter_sequence(&self.mf, &self.ff, &self.bf, &sub, sublen, f1, f3, f3b, self.cm.w as i64,
                    self.std_msvbias_params(do_msvbias, f1b), self.std_vit_params(do_vit, f2, do_vitbias, f2b), do_fwd, do_fwdbias, do_msv, self.maxw)
            };
            // C: each truncated pass runs its own pli_p7_filter → count n_past_msv/fwd
            // (before the F3b-survivor emptiness check, matching C's counting order).
            self.acct.add_f1f3(pass as usize, &pass_f1f3);
            if merged.is_empty() {
                continue;
            }
            let envs = self.trunc_env_def(&sub, sublen, &merged, pass, f4, f4b, do_gfwdbias, f5, do_edefbias, f5b, rt1, rt2, rt3, ns);
            self.cm_trunc_stage(
                &sub, sublen, &envs, pass, local, pli_t, start_offset, f6, gcyk_env_cutoff, do_fcyk,
                do_fcykenv, fcyk_tau, final_tau, do_cyk, do_nonbanded, do_null3, &mut out,
            );
        }

        // ---- Internal ANY pass (PLI_PASS_5P_AND_3P_ANY): --anytrunc/--inttrunc/
        // --onlytrunc. C searches the FULL sequence (sq2search = sq, start_offset = 0,
        // cm_pipeline.c:1449) with do_local_envdef=TRUE: LOCAL p7 domain-def (no glocal
        // F4/F4b filter) then the truncated CM stages (Tcp9 bands, enforce first/final =
        // FALSE so internal truncations are allowed). Reuses whole_merged (the shared
        // full-chunk F1/F3 windows).
        if do_any {
            // C: pass 5 runs pli_p7_filter on the FULL sequence (sq2search = sq),
            // charging its own n_past_msv/fwd/fwdbias over the identical whole-chunk
            // windows — but only now, past the nwin_pass_std_any==0 early return.
            self.acct.add_f1f3(PLI_PASS_5P_AND_3P_ANY as usize, &whole_f1f3);
            // The LOCAL domain-def needs the odds-space Forward parser filter, exactly
            // as the --hmmonly stage (distinct from the F3 window filter self.ff).
            if let Some(p7) = self.cm.p7.as_ref() {
                let ff_fb = crate::p7_fwdback::build_forward_filter(p7);
                let envs = self.pass5_env_def(&ff_fb, wdsq, win_len, &whole_merged, f5, do_edefbias, f5b);
                self.cm_trunc_stage(
                    wdsq, win_len, &envs, PLI_PASS_5P_AND_3P_ANY, local, pli_t, 0, f6,
                    gcyk_env_cutoff, do_fcyk, do_fcykenv, fcyk_tau, final_tau, do_cyk,
                    do_nonbanded, do_null3, &mut out,
                );
            }
        }
        out
    }

    /// C `pli_p7_env_def` (cm_pipeline.c:2935) for one pass: glocal Forward (F4) +
    /// bias (F4b) + glocal envelope definition (F5), using the pass-specific p7
    /// profile (gm / Rgm / Lgm / Tgm). Returns surviving envelope (es,ee) pairs in
    /// `sq2search`-local coords.
    #[allow(clippy::too_many_arguments)]
    fn trunc_env_def(
        &self,
        sub_full: &[u8],
        sublen: usize,
        merged: &[crate::cm_pipeline::Window],
        pass_idx: i32,
        f4: f64,
        f4b: f64,
        do_gfwdbias: bool,
        f5: f64,
        do_edefbias: bool,
        f5b: f64,
        rt1: f32,
        rt2: f32,
        rt3: f32,
        ns: usize,
    ) -> Vec<(i32, i32)> {
        use crate::cm_trunc::{
            PLI_PASS_STD_ANY, PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE,
            PLI_PASS_5P_AND_3P_FORCE,
        };
        // Per-pass profile + p-value stats + score correction (cm_pipeline.c:3000-3005,
        // 3088-3130). use_local_stats selects LFTAU/LFLAMBDA over GFMU/GFLAMBDA.
        // safe_correction: Some(c) for Rgm/Lgm ⇒ F4/F4b use (fwdsc+c - nullsc) and F5
        // adds c/ln2; None for gm/Tgm ⇒ F4=(fwdsc-nullsc), F4b=(fwdsc-filtersc).
        let (mut gm, use_local_stats, safe_correction): (GlocalProfile, bool, Option<f32>) =
            match pass_idx {
                x if x == PLI_PASS_5P_ONLY_FORCE => (self.rgm_proto.clone(), true, Some(0.0)),
                x if x == PLI_PASS_3P_ONLY_FORCE => {
                    (self.lgm_proto.clone(), true, Some((1.0f32 / self.cm.clen as f32).ln()))
                }
                x if x == PLI_PASS_5P_AND_3P_FORCE => (self.tgm_proto.clone(), true, None),
                _ => (self.gm_proto.clone(), false, None),
            };
        let (mu, lambda) = if use_local_stats {
            (self.lftau, self.lflambda)
        } else {
            (self.gfmu, self.gflambda)
        };
        let enforce_first = crate::cp9::cm_pli_pass_enforces_first_res(pass_idx);
        let enforce_final = crate::cp9::cm_pli_pass_enforces_final_res(pass_idx);
        let ln2 = self.ln2;
        let is_tgm = pass_idx == PLI_PASS_5P_AND_3P_FORCE;
        let is_std = pass_idx == PLI_PASS_STD_ANY;

        let mut envs: Vec<(i32, i32)> = Vec::new();
        for w in merged.iter() {
            let ws: i64 = w.start;
            let we: i64 = w.end;
            // C 3055-3056: enforce first/final residue of sq2search per pass.
            if enforce_first && ws != 1 {
                continue;
            }
            if enforce_final && we != sublen as i64 {
                continue;
            }
            let wlen = (we - ws + 1) as usize;
            let mut sub = vec![255u8];
            sub.extend_from_slice(&sub_full[ws as usize..=we as usize]);
            sub.push(255u8);

            let nullsc = p7_bg_null_one(wlen);
            // Length reconfig: Rgm→5', Lgm→3', Tgm→none, gm→standard (C 3101/3117/gm).
            match pass_idx {
                x if x == PLI_PASS_5P_ONLY_FORCE => {
                    crate::p7_generic::p7_reconfig_length_5prime_trunc(&mut gm, wlen as i32)
                }
                x if x == PLI_PASS_3P_ONLY_FORCE => {
                    crate::p7_generic::p7_reconfig_length_3prime_trunc(&mut gm, wlen as i32)
                }
                x if x == PLI_PASS_5P_AND_3P_FORCE => { /* Tgm: no reconfig (C 3089) */ }
                _ => reconfig_length(&mut gm, wlen as i32),
            }
            let mut gx = P7Gmx::new(gm.m, wlen);
            let fwdsc = p7_gforward(&sub, wlen, &gm, &mut gx);

            // F4: glocal Forward P-value (C 3097-3137).
            let safe_lfwdsc = fwdsc + safe_correction.unwrap_or(0.0);
            let sc_f4 = if safe_correction.is_some() {
                (safe_lfwdsc as f64 - nullsc) / ln2
            } else {
                (fwdsc as f64 - nullsc) / ln2
            };
            if esl_exp_surv(sc_f4, mu, lambda) > f4 {
                continue;
            }
            // C 3139: window survived glocal Forward (F4). Counted unconditionally
            // (C has no `if(do_gfwd)` gate on the F4 filter — do_gfwd only drives the
            // stat display); --noF4 therefore leaves the filtering identical.
            self.acct.add_gfwd(pass_idx as usize, wlen as u64);
            // F4b: composition bias — only when do_gfwdbias (C 3146 `if(pli->do_gfwdbias)`;
            // --noF4b skips the sub-filter entirely and its window count).
            if do_gfwdbias {
                let filtersc = bias_filter_score(&self.bf, &sub, wlen);
                let sc_f4b = if safe_correction.is_some() {
                    (safe_lfwdsc as f64 - nullsc) / ln2
                } else {
                    (fwdsc as f64 - filtersc as f64) / ln2
                };
                if esl_exp_surv(sc_f4b, mu, lambda) > f4b {
                    continue;
                }
                // C 3168-3169: window survived glocal Forward bias (F4b).
                self.acct.add_gfwdbias(pass_idx as usize, wlen as u64);
            }
            let _ = is_tgm;
            let _ = is_std;

            // Backward + glocal domain definition (C 3190-3215).
            let mut gxb = P7Gmx::new(gm.m, wlen);
            let _b = p7_gbackward(&sub, wlen, &gm, &mut gxb);
            let do_null2 = false;
            // C save_mode_is_unihit: the truncated-pass profiles (Rgm/Lgm/Tgm) are
            // UNILOCAL/UNIGLOCAL = unihit → their trunc length model must be preserved;
            // only the STD-pass glocal gm is multihit and gets reconfigured to unihit.
            let is_unihit = pass_idx != PLI_PASS_STD_ANY;
            let domains = p7_domaindef_glocal(&mut gm, &sub, wlen, &gx, &gxb, do_null2, rt1, rt2, rt3, ns, is_unihit);
            let omega = 1.0f32 / 256.0f32;
            for dom in domains.iter() {
                // C 3230-3252: envelope score + correction + F5 P-value.
                let env_len = dom.jenv - dom.ienv + 1;
                let env_sc = dom.envsc
                    + (wlen as f32 - env_len as f32) * ((wlen as f32) / (wlen as f32 + 3.0)).ln();
                let env_edefbias = if do_null2 {
                    p7_flogsum(0.0, omega.ln() + dom.domcorrection)
                } else {
                    0.0
                };
                let mut env_sc_for_pvalue = (env_sc as f64 - (nullsc + env_edefbias as f64)) / ln2;
                if let Some(c) = safe_correction {
                    env_sc_for_pvalue += c as f64 / ln2;
                }
                if esl_exp_surv(env_sc_for_pvalue, mu, lambda) <= f5 {
                    // C 3265-3266: envelope survived env definition (F5).
                    self.acct.add_edef(pass_idx as usize, env_len as u64);
                    // F5b: per-envelope composition bias (C 3254-3277) — only when
                    // do_edefbias. Recompute the envelope P-value against the WINDOW bias
                    // (p7_bg_FilterScore), re-adding the same Rgm/Lgm safe_correction.
                    let mut keep = true;
                    if do_edefbias {
                        let filtersc = bias_filter_score(&self.bf, &sub, wlen);
                        let mut env_sc_b = (env_sc as f64 - filtersc as f64) / ln2;
                        if let Some(c) = safe_correction {
                            env_sc_b += c as f64 / ln2;
                        }
                        if esl_exp_surv(env_sc_b, mu, lambda) > f5b {
                            keep = false;
                        } else {
                            self.acct.add_edefbias(pass_idx as usize, env_len as u64);
                        }
                    }
                    if keep {
                        let es = (dom.ienv as i64 + ws - 1) as i32;
                        let ee = (dom.jenv as i64 + ws - 1) as i32;
                        envs.push((es, ee));
                    }
                }
            }
        }
        envs
    }

    /// C `pli_p7_env_def` LOCAL branch (cm_pipeline.c:3061-3071 + 3218-3266) for the
    /// internal PLI_PASS_5P_AND_3P_ANY pass (do_local_envdef=TRUE). Unlike the glocal
    /// [`trunc_env_def`], this uses the OPTIMIZED (local) p7 profile via
    /// `p7_domaindef_ByPosteriorHeuristics` (= [`p7_domaindef_local`], the same engine
    /// --hmmonly uses): there is NO glocal Forward (F4/F4b) pre-filter and NO gfwd/
    /// gfwdbias accounting — the window goes straight to domain definition. The
    /// envelope score correction is identical to the glocal path EXCEPT there is no
    /// Rgm/Lgm term, and significance uses the LOCAL Fwd tail (LFTAU/LFLAMBDA). This
    /// pass enforces neither the first nor the final residue. Returns surviving
    /// envelope (es,ee) pairs in `sub_full`-local coords.
    fn pass5_env_def(
        &self,
        ff: &crate::p7_fwdback::ForwardFilter,
        sub_full: &[u8],
        sublen: usize,
        merged: &[crate::cm_pipeline::Window],
        f5: f64,
        do_edefbias: bool,
        f5b: f64,
    ) -> Vec<(i32, i32)> {
        use crate::cm_trunc::PLI_PASS_5P_AND_3P_ANY;
        // do_local_envdef ⇒ LOCAL Fwd statistics (C 3237-3238).
        let (mu, lambda) = (self.lftau, self.lflambda);
        let ln2 = self.ln2;
        // pli->do_null2 is FALSE for the CM pipeline (matches trunc_env_def).
        let do_null2 = false;
        let omega = 1.0f32 / 256.0f32;
        // Pass 5 enforces neither first nor final residue (C cm_pli_PassEnforces*Res
        // return FALSE for PLI_PASS_5P_AND_3P_ANY); include the checks for fidelity.
        let enforce_first = crate::cp9::cm_pli_pass_enforces_first_res(PLI_PASS_5P_AND_3P_ANY);
        let enforce_final = crate::cp9::cm_pli_pass_enforces_final_res(PLI_PASS_5P_AND_3P_ANY);
        let mut envs: Vec<(i32, i32)> = Vec::new();
        for w in merged.iter() {
            let ws: i64 = w.start;
            let we: i64 = w.end;
            if enforce_first && ws != 1 {
                continue;
            }
            if enforce_final && we != sublen as i64 {
                continue;
            }
            let wlen = (we - ws + 1) as usize;
            // C 3053-3057: window subseq, sentinel-padded frame [255, res.., 255].
            let mut sub = vec![255u8];
            sub.extend_from_slice(&sub_full[ws as usize..=we as usize]);
            sub.push(255u8);
            let nullsc = p7_bg_null_one(wlen);
            // C 3062-3071: p7_oprofile_ReconfigLength + Forward/BackwardParser +
            // p7_domaindef_ByPosteriorHeuristics. p7_domaindef_local performs the length
            // reconfig (multihit length model) + fwd/bck + domain decoding internally.
            let dd = crate::p7_domaindef::p7_domaindef_local(ff, &sub, wlen, do_null2);
            // C 3208-3209: no discrete domains / no envelopes ⇒ skip window.
            if dd.nregions == 0 || dd.nenvelopes == 0 {
                continue;
            }
            for dom in dd.domains.iter() {
                // C 3218-3252: envelope score + correction + F5 P-value (no Rgm/Lgm term).
                let env_len = dom.jenv - dom.ienv + 1;
                let env_sc = dom.envsc
                    + (wlen as f32 - env_len as f32) * ((wlen as f32) / (wlen as f32 + 3.0)).ln();
                let env_edefbias = if do_null2 {
                    p7_flogsum(0.0, omega.ln() + dom.domcorrection)
                } else {
                    0.0
                };
                let env_sc_for_pvalue = (env_sc as f64 - (nullsc + env_edefbias as f64)) / ln2;
                if esl_exp_surv(env_sc_for_pvalue, mu, lambda) <= f5 {
                    // C 3255-3256: envelope survived env definition (F5).
                    self.acct.add_edef(PLI_PASS_5P_AND_3P_ANY as usize, env_len as u64);
                    // F5b: per-envelope bias (C 3254-3277); local path ⇒ no Rgm/Lgm term,
                    // LFTAU/LFLAMBDA (= mu/lambda here).
                    let mut keep = true;
                    if do_edefbias {
                        let filtersc = bias_filter_score(&self.bf, &sub, wlen);
                        let env_sc_b = (env_sc as f64 - filtersc as f64) / ln2;
                        if esl_exp_surv(env_sc_b, mu, lambda) > f5b {
                            keep = false;
                        } else {
                            self.acct.add_edefbias(PLI_PASS_5P_AND_3P_ANY as usize, env_len as u64);
                        }
                    }
                    if keep {
                        let es = (dom.ienv + ws - 1) as i32;
                        let ee = (dom.jenv + ws - 1) as i32;
                        envs.push((es, ee));
                    }
                }
            }
        }
        envs
    }

    /// GLOBAL STANDARD CM stages (F6 CYK filter + F7 Inside final) for the `-g`
    /// STD pass: cp9_global bands + fast_cyk_scan_hb / fast_finside_scan_hb on
    /// cm_global. Appends hits (trunc="no", pass=1 ⇒ ad=None) to `out`.
    #[allow(clippy::too_many_arguments)]
    fn cm_std_stage(
        &self,
        wdsq: &[u8],
        _win_len: usize,
        envs: &[(i32, i32)],
        local: bool,
        pli_t: f32,
        f6: f64,
        gcyk_env_cutoff: f32,
        do_fcyk: bool,
        do_fcykenv: bool,
        fcyk_tau: f64,
        final_tau: f64,
        maxtau: f64,
        do_cyk: bool,
        do_qdb: bool,
        do_nonbanded: bool,
        do_null3: bool,
        out: &mut Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)>,
    ) {
        // GLOBAL (`-g`): global CM + glocal CP9 + GC/GI exp. LOCAL (default): local
        // CM + local CP9(+EL) + LC/LI exp. cm.esc (emission scores → cmcons) and the
        // emit map are config-independent, so cmcons is shared.
        let cm = if local { &self.cm } else { &self.cm_global };
        let cp9 = if local { &self.cp9 } else { &self.cp9_global };
        let map = if local { &self.map } else { &self.map_global };
        let cyk_exp = if local { &self.lc } else { &self.gc };
        // F6 CYK env filter (CYK exp = LC/GC). `--noF6`: skip → envs pass through.
        let mut surv: Vec<(i32, i32)> = Vec::new();
        if !do_fcyk {
            surv = envs.to_vec();
        } else {
        for &(mut es, mut ee) in envs.iter() {
            let (cp9b, _tau, mb) = cp9_iterate_seq2bands(
                cm, cp9, map, wdsq, es, ee, fcyk_tau, maxtau, self.size_limit, true, true,
            );
            if mb > self.size_limit {
                continue;
            }
            let (sc, envi, envj) = fast_cyk_scan_hb(cm, &cp9b, wdsq, es, ee, gcyk_env_cutoff);
            if esl_exp_surv(sc as f64, cyk_exp.mu, cyk_exp.lambda) > f6 {
                continue;
            }
            if do_fcykenv && envi != -1 && envj != -1 {
                es = envi as i32;
                ee = envj as i32;
            }
            // C 3453-3454: envelope survived the CM CYK filter (F6), STD pass.
            self.acct.add_cyk(crate::cm_trunc::PLI_PASS_STD_ANY as usize, (ee - es + 1) as u64);
            surv.push((es, ee));
        }
        }
        // ---- F7 final round: `--qdb`/`--nonbanded` (default config) ----
        // C cm_pipeline.c:705-708 + pli_final_stage (3682-3685): the FINAL Inside round
        // replaces HMM bands with QDB (SMX_QDB2_LOOSE, beta 1e-15) or the full d-range
        // (SMX_NOQDB) — the non-HB integer Inside scanner, per envelope. The F6 CYK
        // filter above is untouched (still HMM-banded). Hits are aligned with CYK-D&C:
        // QDB-banded (dmin2/dmax2) for --qdb, non-banded for --nonbanded (C cm_alndata.c:
        // 384-386 CYKDivideAndConquer, do_qdb ? dmin2 : NULL). This is the same final
        // machinery `nohmm_one_window` uses, on the F5/F6 envelopes rather than windows.
        if do_qdb || do_nonbanded {
            // C: the QDB/nonbanded final Inside round scans cm->smx (CM_SCAN_MX), which is
            // allocated ONLY under --max/--nohmm/--fqdb/--qdb (CM_CONFIG_SCANMX,
            // cm_pipeline.c:736-739). Plain (and -g) --nonbanded never allocate it, so C's
            // pli_dispatch_cm_search returns eslERANGE for every envelope
            // (pli_final_stage 3686-3688: n_overflow_final++, continue) -> 0 CM hits, while
            // the F1-F6 filter stats accumulate normally (verified via instrumented C:
            // smx=(nil), status=eslERANGE, nhit_after=0, ALWAYS, on both STD and truncated
            // passes and every sequence). --max/--nohmm route through their own paths, and
            // --qdb DOES allocate smx and runs the real scan below; only plain/-g
            // --nonbanded reaches here, so we faithfully yield no hits.
            if do_nonbanded {
                return;
            }
            // LOCAL: the local-begin scan CM (finite ibeginsc) with QDB bands. GLOBAL
            // (`-g`, STAGE 2): the global CM. Both carry dmin2/dmax2.
            let qcm = if local { &self.cm_local_scan } else { &self.cm_global };
            let qemap = crate::cp9::create_emit_map(qcm);
            for &(es, ee) in surv.iter() {
                let hits = crate::cm_nohmm::final_stage_inside_opt(
                    qcm, wdsq, &[(es, ee)], pli_t, do_null3, do_nonbanded,
                );
                for h in hits {
                    let lp = h.j - h.i + 1;
                    let subdsq = &wdsq[(h.i as usize - 1)..];
                    // C pli_align_hit -> DispatchSqAlignment (cm_alndata.c:92-101): --qdb
                    // sets CM_ALIGN_SMALL|CYK|QDB, so the display alignment is the BANDED
                    // divide-and-conquer CYKDivideAndConquer(dsq, L, 0,1,L, dmin2,dmax2) —
                    // NOT a flat inside()+insideT(). The QDB2_LOOSE bands (beta=1e-15) are
                    // wide enough that the optimal split path stays in-band; we run the
                    // faithful D&C recursion (generic_splitter) for the exact tie-breaking.
                    let (tr, cyksc) = crate::cm_dpsmall::cyk_divide_and_conquer(qcm, subdsq, lp);
                    let _ = do_nonbanded;
                    let mut ad = crate::cm_alidisplay::cm_alidisplay_create_full(
                        qcm, &self.cmcons, &qemap, &tr, None, subdsq, lp,
                        crate::cm_trunc::PLI_PASS_STD_ANY, true, true, cyksc, 0.0,
                    );
                    // Retain parsetree + subseq for `-A` (approach B); no PP on the
                    // --qdb CYK D&C path (want_pp FALSE, like C DispatchSqAlignment).
                    ad.ali_dsq = Some(sentinel_subdsq(subdsq, lp));
                    ad.ali_tr = Some(tr);
                    out.push((h.i, h.j, h.score, h.bias, ad.cfrom_emit, ad.cto_emit, Some(ad)));
                }
            }
            return;
        }
        // F7 Inside final (global Inside exp = GI); hits above pli_t.
        let emap = crate::cp9::create_emit_map(cm);
        for &(es, ee) in surv.iter() {
            let (cp9b, _tau, mb) = cp9_iterate_seq2bands(
                cm, cp9, map, wdsq, es, ee, final_tau, maxtau, self.size_limit, true, true,
            );
            if mb > self.size_limit {
                continue;
            }
            // Final CM search: Inside (default) or CYK (`--cyk`, C cm_pipeline.c:696
            // clears CM_SEARCH_INSIDE → pli_dispatch_cm_search calls FastCYKScanHB).
            // Same HMM-banded matrix/hit-reporting; only the deck-fill recursion differs.
            let (_sc, _ei, _ej, raw) = if do_cyk {
                crate::cp9::fast_fcyk_scan_hb(cm, &cp9b, wdsq, es, ee, 0.0, pli_t, do_null3)
            } else {
                fast_finside_scan_hb(cm, &cp9b, wdsq, es, ee, 0.0, pli_t, do_null3)
            };
            let surv_hits = remove_overlaps_greedy(raw);
            for (hi, hj, hsc, hbias) in surv_hits {
                // C pli_align_hit (STD pass): cp9_ShiftCMBands (non-trunc) then
                // DispatchSqAlignment -> cm_AlignHB (do_optacc=TRUE, want_pp=TRUE) ->
                // cm_alidisplay_Create. Shifted search bands guarantee the same hit.
                let mut cb = cp9b.clone();
                crate::cp9::shift_cm_bands(cm, &mut cb, hi, hj);
                let lp = hj - hi + 1;
                let hit_dsq = &wdsq[(hi as usize - 1)..];
                let (tr, ppstr, ins_sc, avgpp) =
                    crate::cm_dpalign::cm_align_hb_ad(cm, &cb, hit_dsq, lp);
                let mut ad = crate::cm_alidisplay::cm_alidisplay_create_full(
                    cm, &self.cmcons, &emap, &tr, Some(&ppstr), hit_dsq, lp,
                    crate::cm_trunc::PLI_PASS_STD_ANY, true, true, ins_sc, avgpp,
                );
                let cfrom = ad.cfrom_emit;
                let cto = ad.cto_emit;
                // Retain the search parsetree + PP + subseq for cmsearch `-A` (approach B).
                ad.ali_dsq = Some(sentinel_subdsq(hit_dsq, lp));
                ad.ali_pp = Some(ppstr);
                ad.ali_tr = Some(tr);
                out.push((hi, hj, hsc, hbias, cfrom, cto, Some(ad)));
            }
        }
    }

    /// GLOBAL TRUNCATED CM stages (F6 TrCYKScanHB + F7 FTrInsideScanHB) for a
    /// truncated pass: truncated CP9 bands (Rcp9/Lcp9/Tcp9) + tr_cyk_scan_hb /
    /// ftr_inside_scan_hb on cm_global. mdl bounds + trunc column come from a
    /// non-banded TrCYK alignment of each hit subsequence (the truncated HB
    /// alignment C's pli_align_hit uses is not yet ported; see report). Appends
    /// hits with ad carrying trunc + pass_idx; `start_offset` shifts term3sq coords.
    #[allow(clippy::too_many_arguments)]
    fn cm_trunc_stage(
        &self,
        wdsq: &[u8],
        sublen: usize,
        envs: &[(i32, i32)],
        pass_idx: i32,
        local: bool,
        pli_t: f32,
        start_offset: i64,
        f6: f64,
        gcyk_env_cutoff: f32,
        do_fcyk: bool,
        do_fcykenv: bool,
        fcyk_tau: f64,
        final_tau: f64,
        do_cyk: bool,
        do_nonbanded: bool,
        do_null3: bool,
        out: &mut Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)>,
    ) {
        use crate::cm_trunc::{
            PLI_PASS_5P_ONLY_FORCE, PLI_PASS_3P_ONLY_FORCE, PLI_PASS_5P_AND_3P_FORCE,
        };
        use crate::cm_dpsearch_trunc::{tr_cyk_scan_hb, ftr_inside_scan_hb};
        // GLOBAL (`-g`): global CM + no-EL trunc CP9s + GC/GI exp + g_ptyAA (local=false).
        // LOCAL (default): local CM + EL trunc CP9s + LC/LI exp + l_ptyAA (local=true).
        // The truncation penalties (self.trp) hold both g and l arrays (built once from
        // the un-localized cm.t, C cm_modelconfig.c:274); the scanners/aligners select
        // via the `local` flag (cm_dpsearch_trunc.c:544/2262/3384, cm_trunc pty_slice).
        let cm = if local { &self.cm } else { &self.cm_global };
        let cp9: &CP9 = match pass_idx {
            x if x == PLI_PASS_5P_ONLY_FORCE => if local { &self.rcp9_local } else { &self.rcp9 },
            x if x == PLI_PASS_3P_ONLY_FORCE => if local { &self.lcp9_local } else { &self.lcp9 },
            _ => if local { &self.tcp9_local } else { &self.tcp9 },
        };
        let map = if local { &self.map } else { &self.map_global };
        let cyk_exp = if local { &self.lc } else { &self.gc };
        let emap = crate::cp9::create_emit_map(cm);
        let th1 = DEFAULT_CP9BANDS_THRESH1;
        let th2 = DEFAULT_CP9BANDS_THRESH2;
        let enforce_first = crate::cp9::cm_pli_pass_enforces_first_res(pass_idx);
        let enforce_final = crate::cp9::cm_pli_pass_enforces_final_res(pass_idx);

        // F6 TrCYK env filter (CYK exp = LC/GC). `--noF6`: skip → envs pass through.
        let mut surv: Vec<(i32, i32)> = Vec::new();
        if !do_fcyk {
            surv = envs.to_vec();
        } else {
        for &(mut es, mut ee) in envs.iter() {
            let (cp9b, _pmx) =
                cp9_seq2bands_trunc(cm, cp9, map, &emap, wdsq, es, ee, fcyk_tau, pass_idx, th1, th2);
            let (_hits, vsc, _vmode, envi, envj) = tr_cyk_scan_hb(
                cm, &self.trp, &cp9b, wdsq, es, ee, 0.0, do_null3, pass_idx, gcyk_env_cutoff, local,
            );
            if esl_exp_surv(vsc as f64, cyk_exp.mu, cyk_exp.lambda) > f6 {
                continue;
            }
            // fcykenv envelope refinement, honoring enforce_i0/j0 (C 3435-3438).
            if do_fcykenv && envi != -1 && envj != -1 {
                if !enforce_first {
                    es = envi as i32;
                }
                if !enforce_final {
                    ee = envj as i32;
                }
            }
            // C 3453-3454: envelope survived the CM CYK filter (F6), truncated pass.
            self.acct.add_cyk(pass_idx as usize, (ee - es + 1) as u64);
            surv.push((es, ee));
        }
        }
        // F7 FTrInside final (Inside exp = LI/GI); hits above pli_t.
        // For --nonbanded, C's truncated final round scans cm->trsmx (CM_TR_SCAN_MX),
        // allocated only under --max/--nohmm/--fqdb/--qdb (CM_CONFIG_TRSCANMX,
        // cm_pipeline.c:740). Plain --nonbanded never allocates it, so C's
        // pli_dispatch_cm_search -> RefITrInsideScan returns eslERANGE for every
        // envelope (0 truncated hits), while the F1-F6 truncated filter stats (charged
        // above) are unaffected. --qdb disables truncation entirely, so only plain/-g
        // --nonbanded reaches here.
        if do_nonbanded {
            return;
        }
        for &(es, ee) in surv.iter() {
            let (cp9b, _pmx) =
                cp9_seq2bands_trunc(cm, cp9, map, &emap, wdsq, es, ee, final_tau, pass_idx, th1, th2);
            // Final truncated CM search: FTrInside (default) or TrCYK (`--cyk`,
            // cm_pipeline.c:696 clears CM_SEARCH_INSIDE → pli_dispatch_cm_search calls
            // TrCYKScanHB). Same truncated HMM-banded matrix/hit-reporting tuple.
            let (hits, _vsc, _vmode, _ei, _ej) = if do_cyk {
                tr_cyk_scan_hb(
                    cm, &self.trp, &cp9b, wdsq, es, ee, pli_t, do_null3, pass_idx, 0.0, local,
                )
            } else {
                ftr_inside_scan_hb(
                    cm, &self.trp, &cp9b, wdsq, es, ee, pli_t, do_null3, pass_idx, 0.0, local,
                )
            };
            for h in hits {
                // Full per-hit alignment (mdl bounds + trunc + display + PP) via
                // truncated HMM-banded OPTIMAL-ACCURACY alignment of the hit
                // subsequence, using the search's F7 bands shifted to the hit frame.
                // C: pli_align_hit -> cp9_ShiftCMBands(do_trunc=TRUE) ->
                //   DispatchSqAlignment(CM_ALIGN_TRUNC|HBANDED|OPTACC|POST, hit->mode) ->
                //   cm_TrAlignHB(do_optacc=TRUE): Inside_hb(preset hit->mode) -> Outside_hb
                //   -> Posterior_hb -> EmitterPosterior_hb -> cm_TrOptAccAlignHB + traceback
                //   -> PostCode -> cm_alidisplay_Create. use_local selects l/g_ptyAA.
                let lp = h.j - h.i + 1;
                let hit_dsq = &wdsq[(h.i as usize - 1)..];
                let mut cb = cp9b.clone();
                crate::cp9::shift_cm_bands_trunc(cm, &mut cb, h.i, h.j);
                let trp = &self.trp;
                let (lm, rm) = (&self.lmesc, &self.rmesc);
                let (ins, mode, _isc) = crate::cm_trunc::tr_inside_align_hb(
                    cm, trp, &cb, lm, rm, hit_dsq, lp, pass_idx, h.mode, local,
                );
                let out_mx = crate::cm_trunc::tr_outside_align_hb(
                    cm, trp, &cb, lm, rm, hit_dsq, lp, mode, pass_idx, local, &ins,
                );
                let post = crate::cm_trunc::tr_posterior_hb(cm, &cb, lp, mode, &ins, &out_mx);
                let emit = crate::cm_trunc::tr_emitter_posterior_hb(cm, &cb, lp, mode, &post);
                let (tr, _pp) =
                    crate::cm_trunc::tr_optacc_align_hb(cm, &cb, lp, pass_idx, mode, trp, &emit);
                let (ppstr, avgpp) = crate::cm_trunc::tr_postcode(cm, lp, &emit, &tr);
                // C cm_alidisplay.c:188-189 (via pli_align_hit's cm_alidisplay_Create call,
                // cm_pipeline.c:4369 passes sq2search + seqoffset=hit->start): have_i0/have_j0
                // are per-hit, TRUE only when the hit's aligned subseq starts at position 1
                // (resp. ends at the last position, sq->n) of THIS pass's search subsequence
                // (`sub`, length `sublen`). seqoffset = h.i (hit->start in sub coords),
                // tr.emitr[0] = last emitted residue of the root. These gate the 5P/3P span
                // walk in ParsetreeToCMBounds: a hit that does NOT touch the enforced terminus
                // (e.g. an interior tiny hit at seq 72 of a 124-nt term5sq) must NOT get a
                // guessed truncated span, so it renders non-truncated ("no"), matching C.
                let have_i0 = h.i == 1;
                let have_j0 = (h.i + tr.emitr[0] - 1) == sublen as i32;
                let mut ad = crate::cm_alidisplay::cm_alidisplay_create_full(
                    cm, &self.cmcons, &emap, &tr, Some(&ppstr), hit_dsq, lp,
                    pass_idx, have_i0, have_j0, 0.0, avgpp,
                );
                // Retain the marginal-mode truncated parsetree + PP + subseq for `-A`
                // (approach B). tr.is_std=false/tr.pass_idx drive the allow_trunc
                // missing-char ('~') injection inside parsetrees_to_alignment.
                ad.ali_dsq = Some(sentinel_subdsq(hit_dsq, lp));
                ad.ali_pp = Some(ppstr);
                ad.ali_tr = Some(tr);
                let (cfrom_emit, cto_emit) = (ad.cfrom_emit, ad.cto_emit);
                let hi = (h.i as i64 + start_offset) as i32;
                let hj = (h.j as i64 + start_offset) as i32;
                out.push((hi, hj, h.score, h.bias, cfrom_emit, cto_emit, Some(ad)));
            }
        }
    }

    /// HMM-only pipeline on one chunk × strand (C `pli_final_stage_hmmonly`,
    /// cm_pipeline.c:3844). The CM is never used: the p7 filter cascade produces
    /// windows and `run_hmmonly_stage` runs LOCAL p7 domain-def + scoring per window.
    /// F2 (Viterbi) is not yet wired — running F1+F3 passes a SUPERSET of windows;
    /// extra windows yield no above-`T` domain (→ no hit), so the reported hit SET is
    /// unaffected (only window-count statistics differ; deferred with the F2 owner).
    /// Returns chunk-local hits `(start, stop, dom_score, dom_bias, cfrom_emit,
    /// cto_emit, Some(ad))`; `run_window_task` remaps coords and the LFTAU/LFLAMBDA
    /// exp-tail + nhmmer eZ in `th` give the pvalue/evalue.
    fn hmmonly_one_strand(
        &self,
        wdsq: &[u8],
        win_len: usize,
        th: &Thresh,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        let p7 = match self.cm.p7.as_ref() {
            Some(p) => p,
            None => return Vec::new(),
        };
        // F1 (SSV, incl. MSV bias) + F2 (Viterbi) + F3 (Forward + Forward bias) windows,
        // chunk-local. C cm_pipeline.c:2706-2716: cur_do_vit is ON for HMM-only unless
        // --hmmmax (do_max_hmmonly turns all filters off). Viterbi uses the p7 local
        // Viterbi Gumbel (CM_p7_LVMU/LVLAMBDA).
        let vf;
        let vit = if th.hmm_do_max {
            None
        } else {
            vf = crate::p7_vitfilter::build_vit_filter(p7);
            Some(crate::cm_pipeline::VitParams {
                vf: &vf,
                f2: th.hmm_f2,
                // HMM-only pass: cur_do_vitbias is FALSE (cm_pipeline.c:2523).
                do_vitbias: false,
                f2b: 1.0,
                lvmu: p7.evparam.lvmu,
                lvlambda: p7.evparam.lvlambda,
            })
        };
        // HMM-only pass filter config (cm_pipeline.c:2516-2525): cur_do_fwd =
        // !do_max_hmmonly (so --hmmmax skips Forward and every MSV window survives),
        // cur_do_fwdbias = FALSE always, cur_F3b = 1.0. F1/F3 thresholds are the
        // *_hmmonly values; --hmmmax sets F3_hmmonly = 1.0 (moot when do_fwd is FALSE).
        let (merged, acct) = crate::cm_pipeline::f3_filter_sequence(
            &self.mf, &self.ff, &self.bf, wdsq, win_len,
            th.hmm_f1, th.hmm_f3, 1.0, self.cm.w as i64,
            // HMM-only MSV-bias (cur_do_msvbias = do_bias_hmmonly) is applied via the
            // p7 domain-def null2/bias in the hmmonly final stage, not this window
            // filter (verified byte-identical without an F1b window gate). `None`.
            None, vit,
            /* do_fwd */ !th.hmm_do_max, /* do_fwdbias */ false,
            // C cm_pipeline.c:2514: cur_do_msv = TRUE for the HMM-only pass regardless
            // of pli->do_msv, so this pass always uses MSV window detection.
            /* do_msv */ true, self.maxw,
        );
        // HMM-only pipeline statistics: charge the F1(SSV)/F2(Vit)/F3(Fwd) survivors to
        // the HMM_ONLY_ANY pass (cm_pipeline.c pli_hmmonly_pass_statistics).
        self.acct
            .add_f1f3(crate::cm_trunc::PLI_PASS_HMM_ONLY_ANY as usize, &acct);
        if merged.is_empty() {
            return Vec::new();
        }
        let windows: Vec<(i64, i64)> = merged.iter().map(|w| (w.start, w.end)).collect();
        let params = crate::p7_hmmonly::HmmonlyParams {
            t: th.pli_t,
            max_length: p7.max_length,
            omega: 1.0 / 256.0,
            lftau: self.lftau as f32,
            lflambda: self.lflambda as f32,
            do_null2: th.hmm_do_null2,
            search_mode: true, // CM_SEARCH_SEQS (hit->name set by the caller anyway)
        };
        // The hmmonly domain-def needs the odds-space Forward parser filter
        // (p7_fwdback::ForwardFilter), distinct from the F3 window filter (self.ff).
        let ff_fb = crate::p7_fwdback::build_forward_filter(p7);
        let hits = crate::p7_hmmonly::run_hmmonly_stage(
            &ff_fb,
            p7,
            &self.cm.name,
            self.cm.acc.as_deref().unwrap_or(""),
            self.cm.desc.as_deref().unwrap_or(""),
            self.cm.clen,
            "", "", "",
            wdsq,
            win_len as i64,
            &windows,
            &params,
        );
        hits.into_iter()
            .map(|h| {
                let ad = crate::cm_alidisplay::cm_alidisplay_from_p7(&h.ad);
                (
                    h.start as i32,
                    h.stop as i32,
                    h.score,
                    h.bias,
                    h.ad.cfrom_emit as i32,
                    h.ad.cto_emit as i32,
                    Some(ad),
                )
            })
            .collect()
    }

    /// Run the `-g --nohmm` global CM search on one window (`wdsq` is the sentinel-
    /// padded window frame `[255, res.., 255]`, length `win_len`): whole-window
    /// QDB-CYK filter → envelopes → per-envelope QDB integer Inside. Returns
    /// final-stage hits in window-local coords: (start, stop, score, bias, mdl_from,
    /// mdl_to).
    fn nohmm_one_window(
        &self,
        wdsq: &[u8],
        win_len: usize,
        cyk_cutoff: f32,
        pli_t: f32,
        do_fcyk: bool,
        global: bool,
        do_null3: bool,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        // GLOCAL (`-g`) scans the global CM; the default LOCAL config scans the
        // local-begin scan CM (finite ibeginsc → root local begins in generic_scan).
        let cm = if global { &self.cm_global } else { &self.cm_local_scan };
        // C `--noF6` (do_fcyk=FALSE): skip the CYK seq filter — the whole window
        // becomes a single envelope handed straight to the final stage.
        let envs = if do_fcyk {
            crate::cm_nohmm::cyk_seq_filter(cm, wdsq, win_len as i32, cyk_cutoff, do_null3)
        } else {
            vec![(1i32, win_len as i32)]
        };
        // C pli_cyk_seq_filter (cm_pipeline.c:3599): each envelope surviving the CYK
        // seq filter charges n_past_cyk++ / pos_past_cyk += (ee-es+1) to the STD pass —
        // the "Envelopes passing local CM CYK filter" line. Only when the filter ran
        // (do_fcyk); with --noF6 (or --max) the filter is off and the line shows "(off)".
        if do_fcyk {
            for &(es, ee) in &envs {
                self.acct.add_cyk(crate::cm_trunc::PLI_PASS_STD_ANY as usize, (ee - es + 1) as u64);
            }
        }
        if envs.is_empty() {
            return Vec::new();
        }
        let hits = crate::cm_nohmm::final_stage_inside(cm, wdsq, &envs, pli_t, do_null3);
        // Emit map (node structure only; config-independent) for faithful
        // ParsetreeToCMBounds + EL local-end rendering in cm_alidisplay_Create.
        let emap = crate::cp9::create_emit_map(cm);
        hits.into_iter()
            .map(|h| {
                // Global CYK alignment of the hit subsequence dsq[hi..hj] -> parsetree
                // -> cm_alidisplay + (cfrom_emit, cto_emit). C: pli_align_hit ->
                // DispatchSqAlignment (CM_ALIGN_SMALL|CYK|QDB) -> cm_alidisplay_Create.
                let lp = h.j - h.i + 1;
                let subdsq = &wdsq[(h.i as usize - 1)..]; // subdsq[1..=lp] = residues
                // C: the alidisplay's cyksc column (ad->sc) is the CYK parse score
                // from DispatchSqAlignment, not the Inside hit score h.score.
                // --nohmm aligns QDB-banded (dmin2/dmax2): C cm_alndata.c:384-386
                // CYKDivideAndConquer(..., do_qdb ? dmin2 : NULL, ...).
                let (tr, cyksc) = crate::cm_alidisplay::cyk_align_maybe_banded(
                    cm, subdsq, lp, Some(&cm.dmin2), Some(&cm.dmax2),
                );
                // Full mode-aware builder (J-mode, no PP): renders EL local ends
                // (`*[n]*`) and computes cfrom_span/cto_span via emap, exactly as C.
                let ad = crate::cm_alidisplay::cm_alidisplay_create_full(
                    cm, &self.cmcons, &emap, &tr, None, subdsq, lp,
                    crate::cm_trunc::PLI_PASS_STD_ANY, true, true, cyksc, 0.0,
                );
                let cfrom = ad.cfrom_emit;
                let cto = ad.cto_emit;
                (h.i, h.j, h.score, h.bias, cfrom, cto, Some(ad))
            })
            .collect()
    }

    /// Run the `-g --max` global CM search on one window. C `--max` turns every
    /// filter off (do_msv=do_vit=do_fwd=do_gfwd=do_edef=do_fcyk=FALSE): the whole
    /// window is a single envelope (`es=1, ee=n`, cm_pipeline.c:1670-1674) handed
    /// straight to the final stage, which runs a NON-banded (full d-range, SMX_NOQDB)
    /// global integer Inside scan. E-values use the global Inside params (ECMGI).
    /// Alignment/mdl-bounds via non-banded global CYK (C CM_ALIGN_NONBANDED), the
    /// same `cyk_align_global` machinery the nohmm path uses.
    fn max_one_window(
        &self,
        wdsq: &[u8],
        win_len: usize,
        pli_t: f32,
        global: bool,
        do_null3: bool,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        // GLOCAL (`-g`) scans the global CM; the default LOCAL config scans the
        // local-begin scan CM.
        let cm = if global { &self.cm_global } else { &self.cm_local_scan };
        let align_cm = cm;
        let envs = [(1i32, win_len as i32)];
        let hits =
            crate::cm_nohmm::final_stage_inside_opt(cm, wdsq, &envs, pli_t, do_null3, /*nonbanded=*/ true);
        let emap = crate::cp9::create_emit_map(align_cm);
        hits.into_iter()
            .map(|h| {
                // Non-banded global CYK alignment of dsq[hi..hj] -> parsetree ->
                // cm_alidisplay + (cfrom_emit, cto_emit). Bands don't affect the
                // non-banded aligner, so cm_global's scores/consensus are correct.
                let lp = h.j - h.i + 1;
                let subdsq = &wdsq[(h.i as usize - 1)..];
                let (tr, cyksc) = crate::cm_alidisplay::cyk_align_global(align_cm, subdsq, lp);
                // Full mode-aware builder (J-mode, no PP): renders EL local ends
                // and computes cfrom_span/cto_span via emap, exactly as C.
                let ad = crate::cm_alidisplay::cm_alidisplay_create_full(
                    align_cm, &self.cmcons, &emap, &tr, None, subdsq, lp,
                    crate::cm_trunc::PLI_PASS_STD_ANY, true, true, cyksc, 0.0,
                );
                let cfrom = ad.cfrom_emit;
                let cto = ad.cto_emit;
                (h.i, h.j, h.score, h.bias, cfrom, cto, Some(ad))
            })
            .collect()
    }

}

/// [255, digits.., 255] sentinel-padded digital sequence (full genome frame).
fn digitize_sent(abc: &EslAlphabet, seq: &str) -> Vec<u8> {
    let mut d = vec![255u8];
    d.extend(abc.digitize(&seq.to_uppercase()));
    d.push(255u8);
    d
}

/// GC fraction over the hit span, computed from the digitized, sentinel-padded
/// sequence `full` (1-based: full[p] is genome residue p). Digital codes: A=0,
/// C=1, G=2, U=3, degenerate/other ≥4. Counting codes 1|2 is byte-identical to
/// counting ASCII 'G'/'C' over the same span.
fn hit_gc(full: &[u8], gstart: i64, gstop: i64) -> f64 {
    let a = gstart.min(gstop) as usize;
    let b = gstart.max(gstop) as usize;
    let mut gc = 0usize;
    let n = b - a + 1;
    for &c in &full[a..=b] {
        if c == 1 || c == 2 {
            gc += 1;
        }
    }
    // C computes gc in float32: ad->gc = (act[1]+act[2]) / (float)len.
    if n == 0 { 0.0 } else { (gc as f32 / n as f32) as f64 }
}

/// Global overlap removal: cm_tophits_SortForOverlapRemoval +
/// remove_or_mark_overlaps_one_seq_memeff (per seq_idx + strand group).
fn remove_overlaps_global(hits: &mut [Hit]) {
    // sort: seq_idx asc, in_rc (false first), score desc, start asc
    let mut order: Vec<usize> = (0..hits.len()).collect();
    order.sort_by(|&i, &j| {
        let a = &hits[i];
        let b = &hits[j];
        a.seq_idx
            .cmp(&b.seq_idx)
            .then((a.in_rc as u8).cmp(&(b.in_rc as u8)))
            .then(b.score.partial_cmp(&a.score).unwrap())
            .then(a.start.cmp(&b.start))
            .then(a.stop.cmp(&b.stop)) // total order → deterministic regardless of task order
    });
    // walk groups of equal (seq_idx, in_rc)
    let n = order.len();
    let mut gi = 0;
    while gi < n {
        let mut gj = gi + 1;
        while gj < n
            && hits[order[gj]].seq_idx == hits[order[gi]].seq_idx
            && hits[order[gj]].in_rc == hits[order[gi]].in_rc
        {
            gj += 1;
        }
        // within [gi, gj): higher score first; mark lower-scoring overlappers removed
        for a in gi..gj {
            let ia = order[a];
            if hits[ia].removed {
                continue;
            }
            for b in (a + 1)..gj {
                let ib = order[b];
                if hits[ib].removed {
                    continue;
                }
                let overlap = if !hits[ia].in_rc {
                    // forward: start<stop; overlap unless one entirely before other
                    !(hits[ib].stop < hits[ia].start) && !(hits[ia].stop < hits[ib].start)
                } else {
                    // reverse: start>stop; C test uses start/stop swapped
                    !(hits[ib].start < hits[ia].stop) && !(hits[ia].start < hits[ib].stop)
                };
                if overlap {
                    hits[ib].removed = true;
                }
            }
        }
        gi = gj;
    }
}
