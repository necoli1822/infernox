//! faithful_search — reusable library entry point for the byte-parity cmsearch
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
use crate::cp9_faithful::{
    cm_configure_scores, cm_create_transition_map, cm_expected_state_occupancy,
    cp9_build_and_configure, cp9_build_and_configure_global, cp9_iterate_seq2bands, cp9_map_cm2hmm,
    esl_exp_surv, fast_cyk_scan_hb, fast_finside_scan_hb, remove_overlaps_greedy, CP9, CP9Map,
};
use crate::evalue::ExpParams;
use crate::p7_generic::{
    build_glocal_profile, p7_domaindef_glocal, p7_flogsum, p7_gbackward, p7_gforward,
    reconfig_length, GlocalProfile, P7Gmx,
};
use easel::alphabet::EslAlphabet;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

// Per-stage wall-time accumulators (ns), summed across worker threads. Always
// accumulated (coarse, ~per-task/per-envelope granularity → negligible cost);
// printed only when STAGE_TIMING is set. Run with --cpu 1 for clean attribution.
pub static T_F1F3: AtomicU64 = AtomicU64::new(0); // F1 MSV + F3 Forward + F3b bias
pub static T_F4F5: AtomicU64 = AtomicU64::new(0); // glocal Forward/Backward + envelope def
pub static T_F6BAND: AtomicU64 = AtomicU64::new(0); // F6 CP9 HMM banding (seq2bands)
pub static T_F6CYK: AtomicU64 = AtomicU64::new(0); // F6 banded CYK scan
pub static T_F7BAND: AtomicU64 = AtomicU64::new(0); // F7 CP9 HMM banding (seq2bands)
pub static T_F7INS: AtomicU64 = AtomicU64::new(0); // F7 banded Inside scan + null3
#[inline]
fn stage_add(ctr: &AtomicU64, t: std::time::Instant) {
    ctr.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
    /// Per-hit alignment display (C cm_alidisplay). Populated ONLY on the
    /// `-g --nohmm` path; `None` on the default local path.
    pub alignment: Option<crate::cm_alidisplay::CmAliDisplay>,
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
    removed: bool,
    alignment: Option<crate::cm_alidisplay::CmAliDisplay>,
}

/// Search-time configuration (per-call knobs that don't depend on the model).
#[derive(Clone, Debug)]
pub struct FaithfulConfig {
    /// Search only the top (given) strand, not the reverse complement.
    pub toponly: bool,
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
    /// C `-T <x>`: report hits by minimum bit score `x` instead of by E-value.
    /// `None` = default E-value reporting (E <= `e_report`).
    pub t_cutoff: Option<f32>,
    /// C `--notrunc`: disable truncated (TrCYK) alignment passes. Default `true`
    /// for the LIBRARY (existing faithful_search callers unaffected); the cmsearch
    /// binary flips it to `false` (truncation ON) to match C's `cmsearch` default.
    /// Truncated passes run only in GLOBAL config (`global`), non-max/non-mid.
    pub notrunc: bool,
}

impl Default for FaithfulConfig {
    fn default() -> Self {
        Self {
            toponly: false,
            e_report: 10.0,
            global: false,
            nohmm: false,
            max: false,
            mid: false,
            t_cutoff: None,
            notrunc: true,
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
    /// Global Inside exp-tail params (ECMGI) for nohmm final-stage E-values.
    gi: ExpParams,
    /// Global CYK exp-tail params (ECMGC) for the nohmm CYK filter cutoff.
    gc: ExpParams,
    /// Model consensus display info (C CMConsensus_t), built once from cm_global
    /// for the nohmm per-hit cm_alidisplay.
    cmcons: crate::cm_alidisplay::CmConsensus,
    /// Global-config emit map (C cm->emap) for cm_global, used by the truncated
    /// passes' ParsetreeToCMBounds.
    emap_global: crate::cp9_faithful::EmitMap,
    /// GLOBAL truncated-begin penalty arrays (C trp->g_ptyAA), built once.
    trp: crate::cm_trunc::TrPenalties,
    /// Per-MP marginal left/right emission log-odds (C cm->lmesc/rmesc).
    lmesc: Vec<[f32; 4]>,
    rmesc: Vec<[f32; 4]>,
    cp9: CP9,
    map: CP9Map,
    /// GLOBAL-config CP9 HMM (built from `cm_global`) + its map, for the `-g --mid`
    /// HMM-banded path (which bands + scores the global CM, not the localized one).
    cp9_global: CP9,
    map_global: CP9Map,
    mf: MsvFilter,
    ff: ForwardFilter,
    bf: BiasFilter,
    gm_proto: GlocalProfile,
    // model-derived scalars
    maxw: usize,
    size_limit: f32,
    gfmu: f64,
    gflambda: f64,
    ln2: f64,
    li: ExpParams,
    lc: ExpParams,
    cyk_env_cutoff: f32,
    // F6 / tau constants
    f6: f64,
    fcyk_tau: f64,
    final_tau: f64,
    maxtau: f64,
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
        crate::cm_nohmm::cm_calc_qdb_bands(&mut cm_global, 1e-7, 1e-15, 1e-7)
            .map_err(|e| format!("QDB band calculation failed: {e}"))?;
        let gi = cm_global.exp_params.clone(); // ECMGI (global Inside)
        let gc = cm_global.exp_params_global_cyk.clone(); // ECMGC (global CYK)
        // Model consensus for per-hit cm_alidisplay (built once; model-only).
        let cmcons = crate::cm_alidisplay::create_cm_consensus(&cm_global);

        // Truncated-pass machinery (built once, model-only): emit map, GLOBAL
        // truncation penalties (psi from the global-config CM), marginal emissions.
        let emap_global = crate::cp9_faithful::create_emit_map(&cm_global);
        let psi_trunc = crate::cp9_faithful::cm_expected_state_occupancy(&cm_global);
        let trp = crate::cm_trunc::TrPenalties::new(&cm_global, &emap_global, &psi_trunc);
        let (lmesc, rmesc) = crate::cm_trunc::marginal_emissions(&cm_global);

        // ---- build the GLOBAL CP9 HMM (from cm_global) for the `-g --mid` path ----
        // cm_global's probabilities (e/t/null) are untouched by scoring/QDB above, so
        // the CP9 built here is the faithful global-config HMM.
        let emap_g = crate::cp9_faithful::create_emit_map(&cm_global);
        let map_global = cp9_map_cm2hmm(&cm_global);
        let psi_g = cm_expected_state_occupancy(&cm_global);
        let tmap_g = cm_create_transition_map();
        let cp9_global =
            cp9_build_and_configure_global(&cm_global, &emap_g, &map_global, &psi_g, &tmap_g);

        // ---- build CP9 HMM (from global CM), then configure CM scores for the DP ----
        cm.flags |= (1 << 10) | (1 << 11); // CMH_LOCAL_BEGIN | CMH_LOCAL_END
        let emap = crate::cp9_faithful::create_emit_map(&cm);
        let map = cp9_map_cm2hmm(&cm);
        let psi = cm_expected_state_occupancy(&cm);
        let tmap = cm_create_transition_map();
        let cp9 = cp9_build_and_configure(&cm, &emap, &map, &psi, &tmap);
        cm_configure_scores(&mut cm); // localizes cm.t + builds tsc/esc/oesc/beginsc/endsc

        // ---- p7 filters (built once) ----
        // max_length used for MSV filter reconfig + windowing overlap (= pli->maxW)
        let max_length: usize = ((cm.w as f64).max(1.25 * cm.clen as f64)).ceil() as usize;
        let maxw = max_length;
        let ln2 = std::f64::consts::LN_2;

        let mf = build_msv_filter(&p7, max_length);
        let ff = build_forward_filter(&p7);
        let bf = build_bias_filter(&p7.compo, p7.m as usize);
        let gm_proto = build_glocal_profile(&p7, 100);
        let gfmu = p7.evparam.gfmu;
        let gflambda = p7.evparam.gflambda;

        Ok(Self {
            cm,
            cm_global,
            gi,
            gc,
            cmcons,
            emap_global,
            trp,
            lmesc,
            rmesc,
            cp9,
            map,
            cp9_global,
            map_global,
            mf,
            ff,
            bf,
            gm_proto,
            maxw,
            size_limit,
            gfmu,
            gflambda,
            ln2,
            li,
            lc,
            cyk_env_cutoff,
            f6,
            fcyk_tau,
            final_tau,
            maxtau,
        })
    }

    /// The model name (CM `NAME`), for tblout `query name`.
    pub fn model_name(&self) -> &str {
        &self.cm.name
    }

    /// The model accession (CM `ACC`), for tblout `query accession`; `"-"` if none.
    pub fn model_acc(&self) -> &str {
        self.cm.acc.as_deref().unwrap_or("-")
    }

    /// The underlying configured CM (read-only).
    pub fn cm(&self) -> &CM {
        &self.cm
    }

    /// Run the faithful cmsearch pipeline over `seqs` (each an ASCII residue
    /// string). Returns reported hits (E <= `cfg.e_report`) in C's SortByEvalue
    /// order. `seq_idx` on each hit indexes back into `seqs`.
    pub fn search(&self, seqs: &[&str], cfg: &FaithfulConfig) -> Vec<FaithfulHit> {
        let abc = EslAlphabet::rna();

        // ---- Z (search space) and per-model E-value machinery ----
        let total_res: i64 = seqs.iter().map(|s| s.len() as i64).sum();
        let z: f64 = if cfg.toponly { total_res as f64 } else { (total_res * 2) as f64 };
        let z_mb = z / 1_000_000.0;

        // ---- Z-dependent filter thresholds (cm_pipeline.c:497-560) ----
        let smallx1 = 1e-6_f64;
        let (f1, _do_vit, f3, f3b, f4, f4b, f5): (f64, bool, f64, f64, f64, f64, f64);
        if z_mb >= (20000.0 - smallx1) {
            f1 = 0.06; _do_vit = true; f3 = 0.0002; f3b = 0.0002; f4 = 0.0002; f4b = 0.0002; f5 = 0.0002;
        } else if z_mb >= (2000.0 - smallx1) {
            f1 = 0.15; _do_vit = true; f3 = 0.0002; f3b = 0.0002; f4 = 0.0002; f4b = 0.0002; f5 = 0.0002;
        } else if z_mb >= (200.0 - smallx1) {
            f1 = 0.15; _do_vit = true; f3 = 0.0008; f3b = 0.0008; f4 = 0.0008; f4b = 0.0008; f5 = 0.0008;
        } else if z_mb >= (20.0 - smallx1) {
            f1 = 0.35; _do_vit = true; f3 = 0.003; f3b = 0.003; f4 = 0.003; f4b = 0.003; f5 = 0.003;
        } else if z_mb >= (2.0 - smallx1) {
            f1 = 0.35; _do_vit = false; f3 = 0.005; f3b = 0.005; f4 = 0.005; f4b = 0.005; f5 = 0.005;
        } else {
            f1 = 0.35; _do_vit = false; f3 = 0.02; f3b = 0.02; f4 = 0.02; f4b = 0.02; f5 = 0.02;
        }
        // Viterbi (F2) is not applied by f3_filter_sequence (matches do_vit=FALSE cases).

        let e_report = cfg.e_report;
        let nohmm = cfg.global && cfg.nohmm;
        let do_max = cfg.global && cfg.max;
        let do_mid = cfg.global && cfg.mid;
        // GLOBAL truncated-CYK path (C `cmsearch -g` default, truncation ON): run the
        // non-banded 4-pass TrCYK per short window. Takes precedence over the nohmm/
        // default routing when enabled. Only in global config, non-max/non-mid.
        let do_gtrunc = cfg.global && !cfg.max && !cfg.mid && !cfg.notrunc;
        // All three special global modes score with the GLOBAL exp-tail params.
        let use_global_exp = nohmm || do_max || do_mid;

        // Final-stage exp-tail params + effective dbsize: global CYK (ECMGC) for the
        // truncated-CYK path, global Inside (ECMGI) for nohmm/max/mid, local Inside
        // (ECMLI) for the default path.
        let exp_final: ExpParams = if do_gtrunc {
            self.gc.clone()
        } else if use_global_exp {
            self.gi.clone()
        } else {
            self.li.clone()
        };
        let ez_final: f64 = (z / exp_final.dbsize) * exp_final.nrandhits as f64; // cur_eff_dbsize

        // pli->T: reporting cutoff. C `-T <x>` sets it directly (bit score); else it
        // is the min bit score with E-value <= e_report (E2ScoreGivenExpInfo).
        let pli_t: f32 = match cfg.t_cutoff {
            Some(t) => t,
            None => (exp_final.mu + (e_report / ez_final).ln() / (-1.0 * exp_final.lambda)) as f32,
        };

        // nohmm CYK filter cutoff (C pli_cyk_seq_filter): ECMGC.mu + ln(F6)/(-lambda), F6=1e-4.
        let cyk_cutoff: f32 =
            (self.gc.mu + (1e-4_f64).ln() / (-1.0 * self.gc.lambda)) as f32;

        // Digitize every sequence once (shared read-only across worker threads).
        let fulls: Vec<Vec<u8>> = seqs.iter().map(|s| digitize_sent(&abc, s)).collect();

        // Build the independent (chunk × strand) work list. Chunk windowing exactly
        // mirrors C's ReadWindow (CM_MAX_RESIDUE_COUNT residues, maxw overlap); each
        // task is a self-contained window, so they can run in any order — the final
        // deterministic sort + overlap removal make the result order-independent.
        let do_bot = !cfg.toponly;
        let mut tasks: Vec<(usize, usize, usize, usize, bool)> = Vec::new(); // (seq_idx, wgs, wge, win_len, in_rc)
        for (seq_idx, full) in fulls.iter().enumerate() {
            let l = full.len() - 2; // residue count (full = [255, res.., 255])
            let mut new_start = 1usize;
            let mut first = true;
            while new_start <= l {
                let ctx = if first { 0 } else { self.maxw.min(new_start - 1) };
                let wgs = new_start - ctx;
                let new_count = CM_MAX_RESIDUE_COUNT.min(l - new_start + 1);
                let win_len = ctx + new_count;
                let wge = wgs + win_len - 1;
                tasks.push((seq_idx, wgs, wge, win_len, false));
                if do_bot {
                    tasks.push((seq_idx, wgs, wge, win_len, true));
                }
                new_start = wge + 1;
                first = false;
            }
        }

        // Process all tasks in parallel; collect preserves task order → deterministic.
        let per_task: Vec<Vec<Hit>> = tasks
            .par_iter()
            .map(|&(seq_idx, wgs, wge, win_len, in_rc)| {
                let full = &fulls[seq_idx];
                // Build the sentinel-padded window dsq directly from `full` in a single
                // buffer. For the reverse strand this reverse-complements in place from
                // `full` (byte-identical to a wdsq → revcomp_win path).
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
                let hits = if do_gtrunc {
                    self.global_trunc_one_window(&dsq, win_len, cyk_cutoff, pli_t)
                } else if do_max {
                    self.max_one_window(&dsq, win_len, pli_t)
                } else if nohmm {
                    self.nohmm_one_window(&dsq, win_len, cyk_cutoff, pli_t)
                } else if do_mid {
                    let mut gm = self.gm_proto.clone();
                    // --mid: SSV/Vit off (F1 pass-all), Forward/env/CYK P<=Fmid=0.02.
                    self.mid_one_strand(&mut gm, &dsq, win_len, 0.02, 0.02, 0.02, 0.02, 0.02, pli_t)
                } else {
                    let mut gm = self.gm_proto.clone();
                    self.pipeline_one_strand(
                        &mut gm, &dsq, win_len, f1, f3, f3b, f4, f4b, f5, pli_t,
                    )
                };

                hits.into_iter()
                    .map(|(ws_loc, we_loc, sc, bias, mdl_from, mdl_to, ad)| {
                        let (gstart, gstop) = if in_rc {
                            (wge as i64 - ws_loc as i64 + 1, wge as i64 - we_loc as i64 + 1)
                        } else {
                            (wgs as i64 + ws_loc as i64 - 1, wgs as i64 + we_loc as i64 - 1)
                        };
                        let pvalue = esl_exp_surv(sc as f64, exp_final.mu, exp_final.lambda);
                        let evalue = pvalue * ez_final;
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
                            removed: false,
                            alignment: ad,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut all_hits: Vec<Hit> = per_task.into_iter().flatten().collect();

        // ---- global overlap removal (SortForOverlapRemoval + RemoveOrMarkOverlaps) ----
        remove_overlaps_global(&mut all_hits);
        all_hits.retain(|h| !h.removed);

        // ---- SortByEvalue ----
        all_hits.sort_by(|a, b| {
            a.evalue
                .partial_cmp(&b.evalue)
                .unwrap()
                .then(b.score.partial_cmp(&a.score).unwrap())
                .then(a.seq_idx.cmp(&b.seq_idx))
                .then(a.start.cmp(&b.start))
                .then(a.stop.cmp(&b.stop)) // total order → deterministic regardless of task order
        });

        // ---- Threshold ---- (C `-T <x>`: by bit score; else by E-value)
        all_hits
            .into_iter()
            .filter(|h| match cfg.t_cutoff {
                Some(t) => h.score >= t,
                None => h.evalue <= e_report,
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
                alignment: h.alignment,
            })
            .collect()
    }

    /// GLOBAL truncated-CYK search of one short window (C `cmsearch -g` default,
    /// truncation ON). Runs the STD (J-mode) global CYK plus the three truncated
    /// passes (5P_ONLY, 3P_ONLY, 5P_AND_3P) on the whole window `[1..win_len]`,
    /// keeps the best-scoring parse, and derives trunc + mdl bounds from it via
    /// ParsetreeToCMBounds. Each short sequence is its own envelope (i0=1, j0=L).
    ///
    /// For windows longer than the non-banded TrCYK is practical for, falls back to
    /// the (non-truncated) global nohmm CYK path so genome-scale runs stay tractable.
    fn global_trunc_one_window(
        &self,
        wdsq: &[u8],
        win_len: usize,
        cyk_cutoff: f32,
        pli_t: f32,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        use crate::cm_trunc::{
            inside_score_global, parsetree_to_cm_bounds, tr_cyk_align, tr_inside_score,
            trunc_string, PLI_PASS_3P_ONLY_FORCE, PLI_PASS_5P_AND_3P_FORCE, PLI_PASS_5P_ONLY_FORCE,
            PLI_PASS_STD_ANY,
        };
        // Non-banded TrCYK holds ~6 float + 7 int decks of (L+1)^2 cells per state;
        // memory is O(M·L^2). 300 bounds it to a few hundred MB/window while covering
        // any real tRNA candidate (the tRNAscan-SE consumer feeds short sequences).
        const MAX_TRUNC_L: usize = 300;
        let cm = &self.cm_global;

        if win_len > MAX_TRUNC_L {
            // Non-banded TrCYK is O(M·L^2) memory / O(M·L^3) time; too big here.
            // Fall back to the non-truncated global nohmm CYK path.
            return self.nohmm_one_window(wdsq, win_len, cyk_cutoff, pli_t);
        }

        let l = win_len as i32;

        // --- Pass selection by INSIDE score. tRNAscan-SE runs `cmsearch -g --toponly`
        // (NOT --cyk), i.e. C's default Inside/OptAcc pipeline; the winning pass is the
        // one with the highest Inside score. This can differ from the CYK ranking (e.g.
        // a marginal hit where STD wins under CYK but a 5'/3' pass wins under Inside). ---
        let mut best_pass = PLI_PASS_STD_ANY;
        let mut best_inside = inside_score_global(cm, wdsq, l);
        for &pass in &[
            PLI_PASS_5P_ONLY_FORCE,
            PLI_PASS_3P_ONLY_FORCE,
            PLI_PASS_5P_AND_3P_FORCE,
        ] {
            let isc = tr_inside_score(cm, &self.trp, &self.lmesc, &self.rmesc, wdsq, l, pass);
            if isc > best_inside {
                best_inside = isc;
                best_pass = pass;
            }
        }

        // --- CYK alignment of the WINNING pass: the parsetree we derive
        // mdlfrom/mdlto/trunc from, plus the CYK score we report (comparable to C
        // `--cyk`, which the verify harness references). ---
        let (best_tr, best_sc) = if best_pass == PLI_PASS_STD_ANY {
            crate::cm_alidisplay::cyk_align_global(cm, wdsq, l)
        } else {
            let (tr, sc, _mode) =
                tr_cyk_align(cm, &self.trp, &self.lmesc, &self.rmesc, wdsq, l, best_pass);
            (tr, sc)
        };

        if best_sc < pli_t {
            return Vec::new();
        }

        // Whole-window alignment: first/final residue are always included.
        let (cfrom_span, cto_span, cfrom_emit, cto_emit) =
            parsetree_to_cm_bounds(cm, &self.emap_global, &best_tr, best_pass, true, true);
        let trunc = trunc_string(cfrom_span, cto_span, cfrom_emit, cto_emit).to_string();

        // Alignment display. STD parses are pure J-mode → the existing (J-mode)
        // cm_alidisplay is faithful. For truncated (marginal-mode) parses the J-mode
        // display machinery is not applicable, so we carry the derived bounds/trunc
        // in a minimal CM_ALIDISPLAY (tblout mdl/trunc are read from these fields;
        // the consumer derives N from mdl/clen).
        let ad = if best_pass == PLI_PASS_STD_ANY {
            let mut a = crate::cm_alidisplay::cm_alidisplay_create(cm, &self.cmcons, &best_tr, wdsq);
            a.cfrom_span = cfrom_span;
            a.cto_span = cto_span;
            a.trunc = trunc;
            a.pass_idx = best_pass;
            a
        } else {
            trunc_alidisplay(cfrom_emit, cto_emit, cfrom_span, cto_span, trunc, best_pass, cm.clen)
        };

        vec![(1, l, best_sc, 0.0, cfrom_emit, cto_emit, Some(ad))]
    }

    /// Run LOOP-1 (F1/F3/F3b + F4/F4b/F5) then LOOP-2 (F6 + F7) on one strand of
    /// one window. Returns final-stage hits in window-local coords:
    /// (start, stop, score, bias, mdl_from, mdl_to).
    fn pipeline_one_strand(
        &self,
        gm: &mut GlocalProfile,
        wdsq: &[u8],
        win_len: usize,
        f1: f64,
        f3: f64,
        f3b: f64,
        f4: f64,
        f4b: f64,
        f5: f64,
        pli_t: f32,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        let cm = &self.cm;
        let cp9 = &self.cp9;
        let map = &self.map;
        let cyk_env_cutoff = self.cyk_env_cutoff;
        let fcyk_tau = self.fcyk_tau;
        let final_tau = self.final_tau;
        let maxtau = self.maxtau;
        let size_limit = self.size_limit;
        let gfmu = self.gfmu;
        let gflambda = self.gflambda;
        let ln2 = self.ln2;
        let lc = &self.lc;
        let f6 = self.f6;

        // ---- LOOP-1: F1/F3/F3b merged windows ----
        let _t = std::time::Instant::now();
        let merged = f3_filter_sequence(&self.mf, &self.ff, &self.bf, wdsq, win_len, f1, f3, f3b, cm.w as i64);
        stage_add(&T_F1F3, _t);

        // ---- F4/F4b/F5: envelope definition per surviving window ----
        let _t45 = std::time::Instant::now();
        let mut p7envs: Vec<(i32, i32)> = Vec::new(); // (es, ee) window-local
        for w in merged.iter() {
            let ws = w.start;
            let we = w.end;
            let wlen = (we - ws + 1) as usize;

            let mut sub = vec![255u8];
            sub.extend_from_slice(&wdsq[ws as usize..=we as usize]);
            sub.push(255u8);

            let nullsc = p7_bg_null_one(wlen);
            reconfig_length(gm, wlen as i32);
            let mut gx = P7Gmx::new(gm.m, wlen);
            let fwdsc = p7_gforward(&sub, wlen, gm, &mut gx);

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
            let domains = p7_domaindef_glocal(gm, &sub, wlen, &gx, &gxb, do_null2);
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
        stage_add(&T_F4F5, _t45);
        if p7envs.is_empty() {
            return Vec::new();
        }

        // ---- LOOP-2 / F6: CYK env filter ----
        let mut surv_env: Vec<(i32, i32)> = Vec::new();
        for &(mut es, mut ee) in p7envs.iter() {
            let _tb = std::time::Instant::now();
            let (cp9b, _tau, mb) =
                cp9_iterate_seq2bands(cm, cp9, map, wdsq, es, ee, fcyk_tau, maxtau, size_limit, true, true);
            stage_add(&T_F6BAND, _tb);
            if mb > size_limit {
                continue;
            } // eslERANGE overflow: skip envelope
            let _tc = std::time::Instant::now();
            let (sc, envi, envj) = fast_cyk_scan_hb(cm, &cp9b, wdsq, es, ee, cyk_env_cutoff);
            stage_add(&T_F6CYK, _tc);
            let p = esl_exp_surv(sc as f64, lc.mu, lc.lambda);
            if p > f6 {
                continue;
            }
            // do_fcykenv ON: refine envelope boundaries
            if envi != -1 && envj != -1 {
                es = envi as i32;
                ee = envj as i32;
            }
            surv_env.push((es, ee));
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
            let _tb = std::time::Instant::now();
            let (cp9b, _tau, mb) =
                cp9_iterate_seq2bands(cm, cp9, map, wdsq, es, ee, final_tau, maxtau, size_limit, true, true);
            stage_add(&T_F7BAND, _tb);
            if mb > size_limit {
                continue;
            }
            let _ti = std::time::Instant::now();
            let (_sc, _ei, _ej, raw) =
                fast_finside_scan_hb(cm, &cp9b, wdsq, es, ee, 0.0, pli_t, true);
            stage_add(&T_F7INS, _ti);
            let surv = remove_overlaps_greedy(raw);
            // Per surviving hit: HMM-banded CYK align (shifted bands) -> mdl from/to.
            // C: pli_align_hit -> cp9_ShiftCMBands -> DispatchSqAlignment -> ParsetreeToCMBounds.
            for (hi, hj, hsc, hbias) in surv {
                let mut cb = cp9b.clone();
                crate::cp9_faithful::shift_cm_bands(cm, &mut cb, hi, hj);
                let lp = hj - hi + 1;
                let (cfrom, cto) = crate::cp9_faithful::cyk_align_hb_cmbounds(
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
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        let cm = &self.cm_global;
        let _t = std::time::Instant::now();
        let envs = crate::cm_nohmm::cyk_seq_filter(cm, wdsq, win_len as i32, cyk_cutoff, true);
        stage_add(&T_F6CYK, _t);
        if envs.is_empty() {
            return Vec::new();
        }
        let _ti = std::time::Instant::now();
        let hits = crate::cm_nohmm::final_stage_inside(cm, wdsq, &envs, pli_t, true);
        stage_add(&T_F7INS, _ti);
        hits.into_iter()
            .map(|h| {
                // Global CYK alignment of the hit subsequence dsq[hi..hj] -> parsetree
                // -> cm_alidisplay + (cfrom_emit, cto_emit). C: pli_align_hit ->
                // DispatchSqAlignment (CM_ALIGN_SMALL|CYK|QDB) -> cm_alidisplay_Create.
                let lp = h.j - h.i + 1;
                let subdsq = &wdsq[(h.i as usize - 1)..]; // subdsq[1..=lp] = residues
                let (tr, _sc) = crate::cm_alidisplay::cyk_align_global(cm, subdsq, lp);
                let ad = crate::cm_alidisplay::cm_alidisplay_create(cm, &self.cmcons, &tr, subdsq);
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
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        let cm = &self.cm_global;
        let envs = [(1i32, win_len as i32)];
        let _ti = std::time::Instant::now();
        let hits =
            crate::cm_nohmm::final_stage_inside_opt(cm, wdsq, &envs, pli_t, true, /*nonbanded=*/ true);
        stage_add(&T_F7INS, _ti);
        hits.into_iter()
            .map(|h| {
                // Non-banded global CYK alignment of dsq[hi..hj] -> parsetree ->
                // cm_alidisplay + (cfrom_emit, cto_emit). Bands don't affect the
                // non-banded aligner, so cm_global's scores/consensus are correct.
                let lp = h.j - h.i + 1;
                let subdsq = &wdsq[(h.i as usize - 1)..];
                let (tr, _sc) = crate::cm_alidisplay::cyk_align_global(cm, subdsq, lp);
                let ad = crate::cm_alidisplay::cm_alidisplay_create(cm, &self.cmcons, &tr, subdsq);
                let cfrom = ad.cfrom_emit;
                let cto = ad.cto_emit;
                (h.i, h.j, h.score, h.bias, cfrom, cto, Some(ad))
            })
            .collect()
    }

    /// Run the `-g --mid` pipeline on one strand of one window. C `--mid` keeps the
    /// normal (HMM-banded) pipeline but skips SSV(F1) and Viterbi(F2), running
    /// Forward(F3)+bias, glocal envelope def(F4/F4b/F5), HMM-banded CYK filter(F6),
    /// and an HMM-banded final Inside stage — all in GLOBAL config. This is
    /// identical to [`Self::pipeline_one_strand`] except: (a) the F1/F3/F3b filter
    /// runs with MSV off (tiled 2·maxW windows instead of SSV-merged windows), and
    /// (b) the CM banded stages (F6/F7 + hit alignment) use `cm_global`/`cp9_global`
    /// with the global CYK/Inside cutoffs. `f3..f5` are all `Fmid` (0.02).
    fn mid_one_strand(
        &self,
        gm: &mut GlocalProfile,
        wdsq: &[u8],
        win_len: usize,
        f3: f64,
        f3b: f64,
        f4: f64,
        f4b: f64,
        f5: f64,
        pli_t: f32,
    ) -> Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> {
        let cm = &self.cm_global;
        let cp9 = &self.cp9_global;
        let map = &self.map_global;
        let fcyk_tau = self.fcyk_tau;
        let final_tau = self.final_tau;
        let maxtau = self.maxtau;
        let size_limit = self.size_limit;
        let gfmu = self.gfmu;
        let gflambda = self.gflambda;
        let ln2 = self.ln2;
        // GLOBAL exp-tail params: CYK filter cutoff from ECMGC, F6 P-value from ECMGC.
        let gc = &self.gc;
        let f6 = self.f6;
        let f6env = (f6 * 10.0).min(1.0);
        let cyk_env_cutoff: f32 = (gc.mu + f6env.ln() / (-1.0 * gc.lambda)) as f32;

        // ---- F1/F3/F3b: MSV off. Tile the window into 2·maxW windows with
        // maxW-1 overlap (cm_pipeline.c:2621-2637), then run Forward(F3)+bias(F3b)
        // on each and merge fwd-survivors (cm_pipeline.c:2824-2860). ----
        let _t = std::time::Instant::now();
        let merged = self.mid_p7_forward_windows(wdsq, win_len, f3, f3b);
        stage_add(&T_F1F3, _t);

        // ---- F4/F4b/F5: glocal envelope definition (identical to default path;
        // the glocal p7 profile is independent of CM local/global config) ----
        let _t45 = std::time::Instant::now();
        let mut p7envs: Vec<(i32, i32)> = Vec::new();
        for w in merged.iter() {
            let ws = w.0;
            let we = w.1;
            let wlen = (we - ws + 1) as usize;

            let mut sub = vec![255u8];
            sub.extend_from_slice(&wdsq[ws as usize..=we as usize]);
            sub.push(255u8);

            let nullsc = p7_bg_null_one(wlen);
            reconfig_length(gm, wlen as i32);
            let mut gx = P7Gmx::new(gm.m, wlen);
            let fwdsc = p7_gforward(&sub, wlen, gm, &mut gx);

            let sc = (fwdsc as f64 - nullsc) / ln2;
            if esl_exp_surv(sc, gfmu, gflambda) > f4 {
                continue;
            }
            let filtersc = bias_filter_score(&self.bf, &sub, wlen);
            let sc_b = (fwdsc as f64 - filtersc as f64) / ln2;
            if esl_exp_surv(sc_b, gfmu, gflambda) > f4b {
                continue;
            }
            let mut gxb = P7Gmx::new(gm.m, wlen);
            let _bcksc = p7_gbackward(&sub, wlen, gm, &mut gxb);
            let do_null2 = false;
            let domains = p7_domaindef_glocal(gm, &sub, wlen, &gx, &gxb, do_null2);
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
                    let es = (dom.ienv as i64 + ws as i64 - 1) as i32;
                    let ee = (dom.jenv as i64 + ws as i64 - 1) as i32;
                    p7envs.push((es, ee));
                }
            }
        }
        stage_add(&T_F4F5, _t45);
        if p7envs.is_empty() {
            return Vec::new();
        }

        // ---- LOOP-2 / F6: HMM-banded CYK env filter (GLOBAL cm/cp9) ----
        let mut surv_env: Vec<(i32, i32)> = Vec::new();
        for &(mut es, mut ee) in p7envs.iter() {
            let _tb = std::time::Instant::now();
            let (cp9b, _tau, mb) =
                cp9_iterate_seq2bands(cm, cp9, map, wdsq, es, ee, fcyk_tau, maxtau, size_limit, true, true);
            stage_add(&T_F6BAND, _tb);
            if mb > size_limit {
                continue;
            }
            let _tc = std::time::Instant::now();
            let (sc, envi, envj) = fast_cyk_scan_hb(cm, &cp9b, wdsq, es, ee, cyk_env_cutoff);
            stage_add(&T_F6CYK, _tc);
            let p = esl_exp_surv(sc as f64, gc.mu, gc.lambda);
            if p > f6 {
                continue;
            }
            if envi != -1 && envj != -1 {
                es = envi as i32;
                ee = envj as i32;
            }
            surv_env.push((es, ee));
        }
        if surv_env.is_empty() {
            return Vec::new();
        }

        // ---- F7: HMM-banded final Inside stage (GLOBAL cm/cp9) ----
        let emap = crate::create_emit_map(cm).expect("create_emit_map");
        let mut out: Vec<(i32, i32, f32, f32, i32, i32, Option<crate::cm_alidisplay::CmAliDisplay>)> =
            Vec::new();
        for &(es, ee) in surv_env.iter() {
            let _tb = std::time::Instant::now();
            let (cp9b, _tau, mb) =
                cp9_iterate_seq2bands(cm, cp9, map, wdsq, es, ee, final_tau, maxtau, size_limit, true, true);
            stage_add(&T_F7BAND, _tb);
            if mb > size_limit {
                continue;
            }
            let _ti = std::time::Instant::now();
            let (_sc, _ei, _ej, raw) =
                fast_finside_scan_hb(cm, &cp9b, wdsq, es, ee, 0.0, pli_t, true);
            stage_add(&T_F7INS, _ti);
            let surv = remove_overlaps_greedy(raw);
            for (hi, hj, hsc, hbias) in surv {
                let mut cb = cp9b.clone();
                crate::cp9_faithful::shift_cm_bands(cm, &mut cb, hi, hj);
                let lp = hj - hi + 1;
                let (cfrom, cto) = crate::cp9_faithful::cyk_align_hb_cmbounds(
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

    /// C `--mid` F1/F3/F3b window generation with MSV off: tile the pipeline window
    /// into windows of `2·maxW` with `maxW-1` overlap (cm_pipeline.c:2621-2637), run
    /// Forward(F3) then Forward-bias(F3b) on each (cm_pipeline.c:2752-2787), and
    /// merge overlapping fwd-survivors (cm_pipeline.c:2824-2860). Returns merged
    /// window-local (start,end) coordinate pairs (1-based into `wdsq`).
    fn mid_p7_forward_windows(
        &self,
        wdsq: &[u8],
        win_len: usize,
        f3: f64,
        f3b: f64,
    ) -> Vec<(i32, i32)> {
        let maxw = self.maxw as i64;
        let n = win_len as i64;
        // window tiling (do_msv == FALSE branch)
        let mut nwin = 1i64;
        if n > 2 * maxw {
            nwin += (n - 2 * maxw) / (2 * maxw - (maxw - 1));
            if (n - 2 * maxw) % (2 * maxw - (maxw - 1)) > 0 {
                nwin += 1;
            }
        }
        let mut wins: Vec<(i64, i64)> = Vec::new();
        for i in 0..nwin {
            let ws = 1 + i * (maxw + 1);
            let we = (ws + 2 * maxw - 1).min(n);
            wins.push((ws, we));
        }

        // Forward(F3) + Forward-bias(F3b) per window.
        let ln2 = self.ln2;
        let lftau = self.ff.ftau;
        let lflambda = self.ff.flambda;
        let mut surv: Vec<(i64, i64)> = Vec::new();
        for &(ws, we) in wins.iter() {
            let wlen = (we - ws + 1) as usize;
            let mut sub = vec![255u8];
            sub.extend_from_slice(&wdsq[ws as usize..=we as usize]);
            sub.push(255u8);
            let nullsc = p7_bg_null_one(wlen);

            let fwdsc = crate::cm_pipeline::forward_filter_score(&self.ff, &sub, wlen);
            let wsc = (fwdsc as f64 - nullsc) / ln2;
            let p = esl_exp_surv(wsc, lftau, lflambda);
            if p > f3 {
                continue;
            }
            // F3b: Forward composition-bias filter
            let filtersc = bias_filter_score(&self.bf, &sub, wlen);
            let wsc_b = (fwdsc as f64 - filtersc as f64) / ln2;
            let pb = esl_exp_surv(wsc_b, lftau, lflambda);
            if pb > f3b {
                continue;
            }
            surv.push((ws, we));
        }
        if surv.is_empty() {
            return Vec::new();
        }
        // merge overlapping survivors (adjacent windows whose (we+1) >= next ws)
        surv.sort_by(|a, b| a.0.cmp(&b.0));
        let mut merged: Vec<(i32, i32)> = Vec::new();
        let mut i = 0usize;
        while i < surv.len() {
            let ws = surv[i].0;
            let mut we = surv[i].1;
            let mut j = i + 1;
            while j < surv.len() && (we + 1) >= surv[j].0 {
                we = we.max(surv[j].1);
                j += 1;
            }
            merged.push((ws as i32, we as i32));
            i = j;
        }
        merged
    }
}

/// Minimal CM_ALIDISPLAY for a truncated (marginal-mode) parse: carries the
/// derived model bounds + trunc classification (read for tblout mdl/trunc), and a
/// `model` line showing the C `<[N]*` (5') / `*[N]>` (3') truncated model-span
/// notation (N = positions guessed truncated). Other display lines are left empty
/// (the tRNAscan-SE consumer reads tblout mdl/trunc; N is derivable from mdl/clen).
#[allow(clippy::too_many_arguments)]
fn trunc_alidisplay(
    cfrom_emit: i32,
    cto_emit: i32,
    cfrom_span: i32,
    cto_span: i32,
    trunc: String,
    pass_idx: i32,
    _clen: i32,
) -> crate::cm_alidisplay::CmAliDisplay {
    let mut model = String::new();
    if cfrom_emit != cfrom_span {
        model.push_str(&format!("<[{}]*", cfrom_emit - cfrom_span));
    }
    if cto_emit != cto_span {
        model.push_str(&format!("*[{}]>", cto_span - cto_emit));
    }
    crate::cm_alidisplay::CmAliDisplay {
        aseq: String::new(),
        csline: String::new(),
        ncline: String::new(),
        model,
        mline: String::new(),
        rfline: String::new(),
        cfrom_emit,
        cto_emit,
        cfrom_span,
        cto_span,
        trunc,
        pass_idx,
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
