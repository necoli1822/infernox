// SPDX-License-Identifier: BSD-3-Clause
//! Reusable CM alignment configuration + per-sequence dispatch.
//!
//! Faithful port of the alignment-side of C Infernal `cm_Configure`
//! (cm_modelconfig.c) + `DispatchSqAlignment` / `DispatchSqBlockAlignment`
//! (cm_alndata.c). This is the shared substrate used by both `cmalign` (bin) and
//! `cmbuild --refine` (refine_msa's inner cm->parsetrees step). It was factored
//! out of `bin/cmalign.rs` verbatim; the per-mode DP calls are unchanged.

use crate::cm::CM;
use crate::cm_dpalign::{cm_align, cm_align_hb, cm_align_sample};
use crate::cm_submodel::{build_sub_cm, configure_sub, predict_sub_cm_columns, sub_cm2cm_parsetree};
use crate::cm_trunc::{
    marginal_emissions, tr_cyk_align, tr_cyk_align_hb, tr_emitter_posterior, tr_inside_align,
    tr_optacc_align, tr_outside_align, tr_postcode, tr_posterior, tr_stochastic_parsetree,
    TrPenalties, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN,
};
use crate::cp9::{
    cm_create_transition_map, cm_expected_state_occupancy, cp9_build_and_configure,
    cp9_build_and_configure_global, cp9_build_and_configure_sub, cp9_iterate_seq2bands,
    cp9_map_cm2hmm, create_emit_map, EmitMap, CP9Map, CP9,
};
use crate::parsetree::Parsetree;
use crate::easel::random::EslRandom;

/// Alignment options controlling configuration + dispatch (C: cm->align_opts /
/// cm->config_opts bits, plus cmalign's do_* locals). `do_optacc` is derived
/// (`!do_cyk && !do_sample`) and `do_trunc` is `!do_notrunc`.
#[derive(Clone, Copy)]
pub struct AlnOpts {
    pub do_global: bool,
    pub do_sub: bool,
    pub do_notrunc: bool,
    pub do_nonbanded: bool,
    pub do_cyk: bool,
    pub do_sample: bool,
    pub do_small: bool,
    pub want_pp: bool,
    /// C cm->tau (--tau): HMM-band start tail-loss prob (DEFAULT_TAU = 1e-7).
    pub tau: f64,
    /// C mxsize (--mxsize): max DP matrix size in Mb, drives band tightening
    /// in cp9_IterateSeq2Bands (cmalign default 1024.0, cmbuild --refine 2048.0).
    pub mxsize: f32,
    /// C cm->maxtau (--maxtau, default 0.05): max tau allowed while iteratively
    /// tightening HMM bands to fit the DP matrix under mxsize.
    pub maxtau: f64,
    /// C --fixedtau: when TRUE, do NOT tighten bands (clears CM_ALIGN_XTAU); the
    /// iterate loop computes bands once at start_tau. Default FALSE (do tighten).
    pub do_fixedtau: bool,
}

impl AlnOpts {
    #[inline]
    pub fn do_optacc(&self) -> bool {
        !self.do_cyk && !self.do_sample
    }
    #[inline]
    pub fn do_trunc(&self) -> bool {
        !self.do_notrunc
    }
}

/// Configured per-model alignment state (C: the CP9 HMM(s), CP9 map, truncated
/// CP9 + emit map, and truncation-penalty machinery attached to a configured CM).
pub struct AlnCfg {
    pub cp9: Option<CP9>,
    pub cp9map: Option<CP9Map>,
    pub tcp9: Option<CP9>,
    pub trunc_emap: Option<EmitMap>,
    pub trunc_setup: Option<(TrPenalties, Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

/// C: cm_Configure (cm_modelconfig.c) for the alignment use case. Configures the
/// CM's scores in place and builds the CP9 HMM(s) + truncation machinery used to
/// derive per-sequence bands. Extracted verbatim from `bin/cmalign.rs`.
pub fn configure_for_alignment(cm: &mut CM, opts: &AlnOpts) -> AlnCfg {
    let do_nonbanded = opts.do_nonbanded;
    let do_global = opts.do_global;

    // In GLOBAL (-g) alignment the CM has no local begins/ends. C's cm_Configure
    // leaves CMH_LOCAL_BEGIN|CMH_LOCAL_END clear for a global CM, and the truncated
    // DP selects penalties via `(cm->flags & CMH_LOCAL_BEGIN) ? l_ptyAA : g_ptyAA`
    // (cm_dpalign_trunc.c:4177). infernox's on-disk flag `CM_ELSELF` shares bit 10
    // with `CMH_LOCAL_BEGIN` (cm.rs), so a freshly-read CM (every CM with an EL
    // self-transition) has bit 10 set; without this clear, `use_local` below is
    // wrongly TRUE in global mode and the DP applies the LOCAL truncation penalty
    // (l_ptyAA, ~0.07 bits more negative than g_ptyAA), depressing every -g bit sc.
    // The LOCAL branch (below) sets bits 10|11; global must clear them to match C.
    if do_global {
        cm.flags &= !((1 << 10) | (1 << 11)); // clear CMH_LOCAL_BEGIN | CMH_LOCAL_END
    }
    let do_sub = opts.do_sub;
    let do_notrunc = opts.do_notrunc;

    let mut cp9: Option<CP9> = None;
    let mut cp9map: Option<CP9Map> = None;
    let mut tcp9: Option<CP9> = None;
    let mut trunc_emap: Option<EmitMap> = None;

    if do_nonbanded {
        if do_global {
            crate::cm_nohmm::cm_configure_scores_global(cm);
        } else {
            crate::cp9::cm_configure_scores(cm);
        }
    } else if do_global {
        // Build the global CP9 from the raw (un-scored) model probabilities BEFORE
        // cm_configure_scores_global mutates cm.t (so psi is computed on the same
        // probabilities cm_search uses for its global CP9).
        let emap = create_emit_map(cm);
        let map = cp9_map_cm2hmm(cm);
        let psi = cm_expected_state_occupancy(cm);
        let tmap = cm_create_transition_map();
        // For --sub, cm->cp9 is reconfigured with cp9_sw_config(swentry=swexit=
        // (M-1)/M) so the HMM can predict interior start/end columns for
        // truncated seqs (C cm_modelconfig.c:331-336). Otherwise the plain
        // global cp9.
        cp9 = Some(if do_sub {
            cp9_build_and_configure_sub(cm, &emap, &map, &psi, &tmap)
        } else {
            cp9_build_and_configure_global(cm, &emap, &map, &psi, &tmap)
        });
        if !do_notrunc {
            let (_r, _l, t) = crate::cp9::cp9_build_and_configure_trunc(cm, &emap, &map, &psi, &tmap);
            tcp9 = Some(t);
            trunc_emap = Some(emap);
        }
        cp9map = Some(map);
        crate::cm_nohmm::cm_configure_scores_global(cm);
    } else {
        cm.flags |= (1 << 10) | (1 << 11); // CMH_LOCAL_BEGIN | CMH_LOCAL_END
        let emap = create_emit_map(cm);
        let map = cp9_map_cm2hmm(cm);
        let psi = cm_expected_state_occupancy(cm);
        let tmap = cm_create_transition_map();
        cp9 = Some(cp9_build_and_configure(cm, &emap, &map, &psi, &tmap));
        if !do_notrunc {
            let (_r, _l, t) =
                crate::cp9::cp9_build_and_configure_trunc_local(cm, &emap, &map, &psi, &tmap);
            tcp9 = Some(t);
            trunc_emap = Some(emap);
        }
        crate::cp9::cm_configure_scores(cm);
        cp9map = Some(map);
    }

    // Truncated-alignment default (C cmalign: PLI_PASS_5P_AND_3P_FORCE +
    // CM_ALIGN_TRUNC). Build truncation-penalty machinery + marginal emissions.
    let do_trunc = !do_notrunc;
    let trunc_setup: Option<(TrPenalties, Vec<Vec<f32>>, Vec<Vec<f32>>)> = if do_trunc {
        let emap = create_emit_map(cm);
        let psi = cm_expected_state_occupancy(cm);
        let trp = TrPenalties::new(cm, &emap, &psi);
        let (lm, rm) = marginal_emissions(cm);
        Some((trp, lm, rm))
    } else {
        None
    };

    AlnCfg {
        cp9,
        cp9map,
        tcp9,
        trunc_emap,
        trunc_setup,
    }
}

/// Per-sequence alignment result, mirroring the fields of C `CM_ALNDATA`
/// (cm_alndata.c) that cmalign's `output_scores()` (cmalign.c:1978) reports.
/// `avg_pp` is C `data->pp` (average posterior; only meaningful when want_pp);
/// `mb_tot` is C `data->mb_tot` (total DP matrix size in Mb, from
/// `cm_*AlignSizeNeeded*` — deterministic, printed in the "mem (Mb)" column).
pub struct SqAlnResult {
    pub tr: Parsetree,
    pub ppstr: Option<Vec<u8>>,
    pub sc: f32,
    pub avg_pp: f32,
    pub mb_tot: f32,
}

/// C: DispatchSqAlignment (cm_alndata.c:286) — align one sequence <dsq[1..l]> to
/// the configured CM, selecting the DP engine from <opts>. Returns
/// (parsetree, optional PP string, score). Thin wrapper over
/// `dispatch_sq_alignment_data` for callers (cmbuild --refine) that only need
/// the parsetree/PP/score triple.
pub fn dispatch_sq_alignment(
    cm: &CM,
    cfg: &AlnCfg,
    opts: &AlnOpts,
    dsq: &[u8],
    l: i32,
    rng: &mut EslRandom,
) -> (Parsetree, Option<Vec<u8>>, f32) {
    let r = dispatch_sq_alignment_data(cm, cfg, opts, dsq, l, rng);
    (r.tr, r.ppstr, r.sc)
}

/// C: DispatchSqAlignment (cm_alndata.c:286) — full form, also computing the
/// per-sequence average PP (`data->pp`) and total DP matrix size (`data->mb_tot`)
/// that cmalign's score report needs. Extracted verbatim from the per-seq loop of
/// `bin/cmalign.rs`; `rng` is the shared serial RNG (only used by --sample/--gibbs).
pub fn dispatch_sq_alignment_data(
    cm: &CM,
    cfg: &AlnCfg,
    opts: &AlnOpts,
    dsq: &[u8],
    l: i32,
    rng: &mut EslRandom,
) -> SqAlnResult {
    let do_optacc = opts.do_optacc();
    let want_pp = opts.want_pp;
    let do_trunc = opts.do_trunc();
    let do_nonbanded = opts.do_nonbanded;
    let do_cyk = opts.do_cyk;
    let do_sample = opts.do_sample;
    let do_small = opts.do_small;
    let do_sub = opts.do_sub;

    // C data->pp (avg posterior) and data->mb_tot (total DP matrix Mb, from
    // cm_*AlignSizeNeeded*). Set inside the branch that actually runs; 0 for the
    // sample/small/sub paths (their mem/pp columns are outside the -o/--sfile
    // verification scope). cp9_m == cm->cp9->M for SizeNeededCP9Matrix.
    let mut avg_pp: f32 = 0.0;
    let mut mb_tot: f32 = 0.0;
    let cp9_m: i32 = cfg.cp9.as_ref().map(|c| c.m).unwrap_or(0);

    let (tr, ppstr, sc) = if do_sub {
        // C: DispatchSqAlignment do_sub path (cm_alndata.c:370-456).
        let orig_cp9 = cfg.cp9.as_ref().unwrap();
        let (spos, epos) = predict_sub_cm_columns(orig_cp9, dsq, l);
        let (mut sub_cm, submap) = build_sub_cm(cm, spos, epos);
        let (sub_cp9, sub_cp9map) = configure_sub(&mut sub_cm, cm, orig_cp9, &submap);
        let (cp9b, _tau, _mb) = cp9_iterate_seq2bands(
            &sub_cm, &sub_cp9, &sub_cp9map, dsq, 1, l, opts.tau, opts.maxtau, opts.mxsize, false,
            !opts.do_fixedtau,
        );
        // C DispatchSqAlignment do_sub: cm_AlignHB fills pp (data->pp = avg). This
        // sub path's mem/pp columns are outside the -o/--sfile verification scope,
        // but thread avgpp for fidelity with C's data->pp assignment.
        let (sub_tr, pp, sc, avg) = cm_align_hb(&sub_cm, &cp9b, dsq, l, do_optacc, want_pp);
        if want_pp { avg_pp = avg; }
        let full_tr = sub_cm2cm_parsetree(cm, &sub_cm, &sub_tr, &submap);
        (full_tr, pp, sc)
    } else if do_small && do_trunc {
        // C DispatchSqAlignment do_small-first precedence (cm_alndata.c:377-379):
        // if(do_small){ if(do_trunc) TrCYK_DnC(...) else CYKDivideAndConquer(...) }.
        // Truncated divide-and-conquer. Reached by `cmbuild --refine --nonbanded`
        // (default, truncated); cmalign --small forces --notrunc so never hits here.
        // do_post && do_small is incompatible (want_pp is false for refine), so no PP.
        let (trp, lm, rm) = cfg.trunc_setup.as_ref().unwrap();
        let use_local = cm.flags & (1 << 10) != 0;
        let (tr, sc, _mode) =
            crate::truncyk::tr_cyk_dnc(cm, trp, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, use_local);
        (tr, None, sc)
    } else if do_trunc && do_nonbanded && do_cyk {
        let (trp, lm, rm) = cfg.trunc_setup.as_ref().unwrap();
        let use_local = cm.flags & (1 << 10) != 0;
        let (tr, sc, _mode) =
            tr_cyk_align(cm, trp, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, use_local);
        if want_pp {
            let (ins, mode, _isc) = tr_inside_align(
                cm, trp, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local,
            );
            let out = tr_outside_align(
                cm, trp, lm, rm, dsq, l, mode, PLI_PASS_5P_AND_3P_FORCE, use_local, &ins,
            );
            let post = tr_posterior(cm, l, mode, &ins, &out);
            let emit = tr_emitter_posterior(cm, l, mode, &post);
            let (ppstr, _avg) = tr_postcode(cm, l, &emit, &tr);
            (tr, Some(ppstr), sc)
        } else {
            (tr, None, sc)
        }
    } else if do_trunc && do_nonbanded && do_optacc {
        let (trp, lm, rm) = cfg.trunc_setup.as_ref().unwrap();
        let use_local = cm.flags & (1 << 10) != 0;
        // C cm_TrAlign(do_optacc): ret_sc = ins_sc (the truncated Inside score).
        let (ins, mode, isc) = tr_inside_align(
            cm, trp, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local,
        );
        let out = tr_outside_align(
            cm, trp, lm, rm, dsq, l, mode, PLI_PASS_5P_AND_3P_FORCE, use_local, &ins,
        );
        let post = tr_posterior(cm, l, mode, &ins, &out);
        let emit = tr_emitter_posterior(cm, l, mode, &post);
        let (tr, _pp) = tr_optacc_align(cm, l, mode, PLI_PASS_5P_AND_3P_FORCE, trp, &emit);
        if want_pp {
            let (ppstr, _avg) = tr_postcode(cm, l, &emit, &tr);
            (tr, Some(ppstr), isc)
        } else {
            (tr, None, isc)
        }
    } else if do_trunc && !do_nonbanded && do_cyk {
        let (trp, lm, rm) = cfg.trunc_setup.as_ref().unwrap();
        let use_local = cm.flags & (1 << 10) != 0;
        let (cp9b, _pmx, _tau) = crate::cp9::cp9_iterate_seq2bands_trunc_align(
            cm,
            cfg.tcp9.as_ref().unwrap(),
            cfg.cp9map.as_ref().unwrap(),
            cfg.trunc_emap.as_ref().unwrap(),
            dsq,
            1,
            l,
            opts.tau,
            opts.maxtau,
            opts.mxsize,
            PLI_PASS_5P_AND_3P_FORCE,
            !opts.do_fixedtau,
        );
        // C cm_TrAlignSizeNeededHB(&mb_tot) before cm_TrAlignHB (cm_alndata.c:432).
        mb_tot = crate::cm_align_size::cm_tr_align_size_needed_hb(
            cm, &cp9b, cp9_m, l, want_pp, false,
        );
        let (tr, sc, _mode) = tr_cyk_align_hb(
            cm, trp, &cp9b, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local,
        );
        if want_pp {
            let (ins, mode, _isc) = crate::cm_trunc::tr_inside_align_hb(
                cm, trp, &cp9b, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local,
            );
            let out = crate::cm_trunc::tr_outside_align_hb(
                cm, trp, &cp9b, lm, rm, dsq, l, mode, PLI_PASS_5P_AND_3P_FORCE, use_local, &ins,
            );
            let post = crate::cm_trunc::tr_posterior_hb(cm, &cp9b, l, mode, &ins, &out);
            let emit = crate::cm_trunc::tr_emitter_posterior_hb(cm, &cp9b, l, mode, &post);
            let (ppstr, avg) = tr_postcode(cm, l, &emit, &tr);
            avg_pp = avg; // C data->pp
            (tr, Some(ppstr), sc)
        } else {
            (tr, None, sc)
        }
    } else if do_trunc && !do_nonbanded && do_optacc {
        let (trp, lm, rm) = cfg.trunc_setup.as_ref().unwrap();
        let use_local = cm.flags & (1 << 10) != 0;
        let (cp9b, _pmx, _tau) = crate::cp9::cp9_iterate_seq2bands_trunc_align(
            cm,
            cfg.tcp9.as_ref().unwrap(),
            cfg.cp9map.as_ref().unwrap(),
            cfg.trunc_emap.as_ref().unwrap(),
            dsq,
            1,
            l,
            opts.tau,
            opts.maxtau,
            opts.mxsize,
            PLI_PASS_5P_AND_3P_FORCE,
            !opts.do_fixedtau,
        );
        // C cm_TrAlignSizeNeededHB(&mb_tot) before cm_TrAlignHB (cm_alndata.c:432).
        mb_tot = crate::cm_align_size::cm_tr_align_size_needed_hb(
            cm, &cp9b, cp9_m, l, want_pp, false,
        );
        // C cm_TrAlignHB(do_optacc): ret_sc = ins_sc (truncated Inside score).
        let (ins, mode, isc) = crate::cm_trunc::tr_inside_align_hb(
            cm, trp, &cp9b, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local,
        );
        if mode == TRMODE_UNKNOWN {
            // eslEAMBIGUOUS failover: HB standard (non-truncated) alignment.
            let (cp9b_std, _tau2, _mb) = cp9_iterate_seq2bands(
                cm,
                cfg.cp9.as_ref().unwrap(),
                cfg.cp9map.as_ref().unwrap(),
                dsq,
                1,
                l,
                opts.tau,
                opts.maxtau,
                opts.mxsize,
                false,
                !opts.do_fixedtau,
            );
            // eslEAMBIGUOUS failover to non-truncated HB align (C cm_TrAlignHB).
            let (t, p, s, avg) = cm_align_hb(cm, &cp9b_std, dsq, l, do_optacc, want_pp);
            if want_pp { avg_pp = avg; }
            (t, p, s)
        } else {
            let out = crate::cm_trunc::tr_outside_align_hb(
                cm, trp, &cp9b, lm, rm, dsq, l, mode, PLI_PASS_5P_AND_3P_FORCE, use_local, &ins,
            );
            let post = crate::cm_trunc::tr_posterior_hb(cm, &cp9b, l, mode, &ins, &out);
            let emit = crate::cm_trunc::tr_emitter_posterior_hb(cm, &cp9b, l, mode, &post);
            let (tr, _pp) = crate::cm_trunc::tr_optacc_align_hb(
                cm, &cp9b, l, PLI_PASS_5P_AND_3P_FORCE, mode, trp, &emit,
            );
            if want_pp {
                let (ppstr, avg) = tr_postcode(cm, l, &emit, &tr);
                avg_pp = avg; // C data->pp
                (tr, Some(ppstr), isc)
            } else {
                (tr, None, isc)
            }
        }
    } else if do_trunc && !do_nonbanded && do_sample {
        // C cm_TrAlignHB(do_sample) (cm_dpalign_trunc.c:1146). Fill truncated HB
        // Inside, then sample a parsetree AND its marginal mode via
        // cm_TrStochasticParsetreeHB. When have_ppstr (want_pp), C's do_post path
        // then runs TrOutside->TrPosterior->TrEmitterPosterior->TrPostCode using
        // the *sampled* mode (not the Inside max mode) to annotate #=GR PP.
        // ret_sc = sampled-parse score (fsc). (Also used by cmbuild --refine --gibbs.)
        let (trp, lm, rm) = cfg.trunc_setup.as_ref().unwrap();
        let use_local = cm.flags & (1 << 10) != 0;
        let (cp9b, _pmx, _tau) = crate::cp9::cp9_iterate_seq2bands_trunc_align(
            cm,
            cfg.tcp9.as_ref().unwrap(),
            cfg.cp9map.as_ref().unwrap(),
            cfg.trunc_emap.as_ref().unwrap(),
            dsq,
            1,
            l,
            opts.tau,
            opts.maxtau,
            opts.mxsize,
            PLI_PASS_5P_AND_3P_FORCE,
            !opts.do_fixedtau,
        );
        // C cm_TrAlignSizeNeededHB(do_sample) (cm_alndata.c) for the -o/--sfile mem column.
        mb_tot = crate::cm_align_size::cm_tr_align_size_needed_hb(
            cm, &cp9b, cp9_m, l, want_pp, true,
        );
        // C cm_TrInsideAlignHB (mode returned here is overwritten by the sample).
        let (ins, _mode, _isc) = crate::cm_trunc::tr_inside_align_hb(
            cm, trp, &cp9b, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local,
        );
        // C cm_TrStochasticParsetreeHB: samples the parsetree and the marginal mode.
        let (tr, smode, fsc) = crate::cm_trunc::tr_stochastic_parsetree_hb(
            cm, trp, &cp9b, &ins, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN,
            use_local, rng,
        );
        if want_pp {
            // C do_post: Outside/Posterior/Emitter/PostCode using the sampled mode.
            let out = crate::cm_trunc::tr_outside_align_hb(
                cm, trp, &cp9b, lm, rm, dsq, l, smode, PLI_PASS_5P_AND_3P_FORCE, use_local, &ins,
            );
            let post = crate::cm_trunc::tr_posterior_hb(cm, &cp9b, l, smode, &ins, &out);
            let emit = crate::cm_trunc::tr_emitter_posterior_hb(cm, &cp9b, l, smode, &post);
            let (ppstr, avg) = tr_postcode(cm, l, &emit, &tr);
            avg_pp = avg; // C data->pp
            (tr, Some(ppstr), fsc)
        } else {
            (tr, None, fsc)
        }
    } else if !do_trunc && !do_nonbanded && do_sample {
        // C cm_AlignHB(do_sample) (cm_dpalign.c:740): fill non-truncated HB Inside,
        // sample a parsetree via cm_StochasticParsetreeHB, then (if want_pp) run the
        // do_post chain for the #=GR PP string. Used by `cmalign --sample --notrunc`
        // and cmbuild --refine --gibbs --notrunc.
        let (cp9b, _tau, _mb) = cp9_iterate_seq2bands(
            cm,
            cfg.cp9.as_ref().unwrap(),
            cfg.cp9map.as_ref().unwrap(),
            dsq,
            1,
            l,
            opts.tau,
            opts.maxtau,
            opts.mxsize,
            false,
            !opts.do_fixedtau,
        );
        // C cm_AlignSizeNeededHB(do_sample) for the -o/--sfile mem column (no shadow mx).
        mb_tot = crate::cm_align_size::cm_align_size_needed_hb(
            cm, &cp9b, cp9_m, l, want_pp, true,
        );
        let (tr, pp, fsc, avg) = crate::cm_dpalign::cm_align_sample_hb(cm, &cp9b, dsq, l, want_pp, rng);
        if want_pp { avg_pp = avg; }
        (tr, pp, fsc)
    } else if do_trunc && do_nonbanded && do_sample {
        // C cm_TrAlign(do_sample) (cm_dpalign_trunc.c): fill non-banded truncated
        // Inside, sample a parsetree AND its marginal mode via cm_TrStochasticParsetree,
        // then (if want_pp) run the do_post chain Outside/Posterior/EmitterPosterior/
        // TrPostCode with the *sampled* mode for the #=GR PP string.
        let (trp, lm, rm) = cfg.trunc_setup.as_ref().unwrap();
        let use_local = cm.flags & (1 << 10) != 0;
        // C cm_TrInsideAlign (mode returned here is overwritten by the sample).
        let (ins, _mode, _isc) = tr_inside_align(
            cm, trp, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local,
        );
        let (tr, smode, fsc) = tr_stochastic_parsetree(
            cm, trp, &ins, lm, rm, dsq, l, PLI_PASS_5P_AND_3P_FORCE, TRMODE_UNKNOWN, use_local, rng,
        );
        if want_pp {
            let out = tr_outside_align(
                cm, trp, lm, rm, dsq, l, smode, PLI_PASS_5P_AND_3P_FORCE, use_local, &ins,
            );
            let post = tr_posterior(cm, l, smode, &ins, &out);
            let emit = tr_emitter_posterior(cm, l, smode, &post);
            let (ppstr, avg) = tr_postcode(cm, l, &emit, &tr);
            avg_pp = avg;
            (tr, Some(ppstr), fsc)
        } else {
            (tr, None, fsc)
        }
    } else if do_sample {
        cm_align_sample(cm, dsq, l, want_pp, rng)
    } else if do_small {
        let (tr, sc) = crate::cm_dpsmall::cyk_divide_and_conquer(cm, dsq, l);
        (tr, None, sc)
    } else if do_nonbanded {
        cm_align(cm, dsq, l, do_optacc, want_pp)
    } else {
        let (cp9b, _tau, _mb) = cp9_iterate_seq2bands(
            cm,
            cfg.cp9.as_ref().unwrap(),
            cfg.cp9map.as_ref().unwrap(),
            dsq,
            1,
            l,
            opts.tau,
            opts.maxtau,
            opts.mxsize,
            false,
            !opts.do_fixedtau,
        );
        // C cm_AlignSizeNeededHB(&mb_tot) before cm_AlignHB (cm_alndata.c:433).
        mb_tot = crate::cm_align_size::cm_align_size_needed_hb(
            cm, &cp9b, cp9_m, l, want_pp, false,
        );
        // C cm_AlignHB returns ret_avgpp; DispatchSqAlignment sets data->pp = (do_post)
        // ? pp : 0 (cm_alndata.c:471). Thread it so the -o/--sfile "avg pp" column
        // matches on the standard (--notrunc) HB path.
        let (t, p, s, avg) = cm_align_hb(cm, &cp9b, dsq, l, do_optacc, want_pp);
        if want_pp { avg_pp = avg; }
        (t, p, s)
    };

    SqAlnResult { tr, ppstr, sc, avg_pp, mb_tot }
}

/// CM consensus-position boundaries spanned by a parsetree. C: the six RETURN
/// values of ParsetreeToCMBounds (cm_parsetree.c:2611).
#[derive(Debug, Clone, Copy)]
pub struct CmBounds {
    pub cfrom_span: i32,
    pub cto_span: i32,
    pub cfrom_emit: i32,
    pub cto_emit: i32,
    pub first_emit: i32,
    pub final_emit: i32,
}

// C StateLeftDelta/StateRightDelta (cm.c): 1 if the state emits on that side.
#[inline]
fn bounds_sdl(stt: i32) -> i32 {
    use crate::constants::{IL_ST, ML_ST, MP_ST};
    if stt == MP_ST || stt == ML_ST || stt == IL_ST { 1 } else { 0 }
}
#[inline]
fn bounds_sdr(stt: i32) -> i32 {
    use crate::constants::{IR_ST, MP_ST, MR_ST};
    if stt == MP_ST || stt == MR_ST || stt == IR_ST { 1 } else { 0 }
}
// C ModeEmitsLeft/ModeEmitsRight (cm.h): J and L emit left; J and R emit right.
#[inline]
fn bounds_mode_emits_left(mode: i8) -> bool {
    use crate::cm_trunc::{TRMODE_J, TRMODE_L};
    mode == TRMODE_J as i8 || mode == TRMODE_L as i8
}
#[inline]
fn bounds_mode_emits_right(mode: i8) -> bool {
    use crate::cm_trunc::{TRMODE_J, TRMODE_R};
    mode == TRMODE_J as i8 || mode == TRMODE_R as i8
}

/// C: ParsetreeToCMBounds() (cm_parsetree.c:2611). Determine the CM consensus
/// position boundaries spanned in a parsetree. `have_i0`/`have_j0` are TRUE if the
/// first/final residue of the source sequence is emitted in `tr` (cmalign passes
/// TRUE, TRUE — cm_alndata.c:462, which reads first_emit/final_emit as spos/epos).
pub fn parsetree_to_cm_bounds(cm: &CM, tr: &Parsetree, have_i0: bool, have_j0: bool) -> CmBounds {
    use crate::constants::{IL_ST, IR_ST, MATL_nd, MATP_nd, MATR_nd};
    let emap = create_emit_map(cm);
    let m = cm.m;

    // cfrom_span = cfrom_emit = first_emit = clen+1; cto_span = cto_emit = final_emit = 0;
    let mut cfrom_emit = emap.clen + 1;
    let mut first_emit = emap.clen + 1;
    let mut cto_emit = 0i32;
    let mut final_emit = 0i32;
    let mut cfrom_span; // set at the end
    let mut cto_span;

    for ti in 0..tr.n as usize {
        let mut v = tr.state[ti];
        let mut mode = tr.mode[ti];
        // C: sdl=StateLeftDelta(cm->sttype[v]) with v possibly == cm->M (EL sentinel);
        // C's sttype array has an M-index sentinel, but here sttype has no index m, and
        // for v==m these are overwritten to 0 in the else branch below (value discarded),
        // so compute 0 when v==m to avoid the out-of-bounds read.
        let mut sdl = if v != m { bounds_sdl(cm.sttype[v as usize] as i32) } else { 0 };
        let mut sdr = if v != m { bounds_sdr(cm.sttype[v as usize] as i32) } else { 0 };
        let nd;
        let lpos;
        let rpos;
        let is_left;
        let is_right;
        let mut insert_sd = 0i32;
        if v != m {
            nd = cm.ndidx[v as usize] as usize;
            let ndt = cm.ndtype[nd] as i32;
            lpos = if ndt == MATP_nd || ndt == MATL_nd { emap.lpos[nd] } else { emap.lpos[nd] + 1 };
            rpos = if ndt == MATP_nd || ndt == MATR_nd { emap.rpos[nd] } else { emap.rpos[nd] - 1 };
            let stt = cm.sttype[v as usize] as i32;
            if stt == IL_ST {
                is_left = true; is_right = false; insert_sd = 1;
            } else if stt == IR_ST {
                is_left = false; is_right = true; insert_sd = 1;
            } else if ndt == MATP_nd {
                is_left = true; is_right = true; insert_sd = 0;
            } else if ndt == MATL_nd {
                is_left = true; is_right = false; insert_sd = 0;
            } else if ndt == MATR_nd {
                is_left = false; is_right = true; insert_sd = 0;
            } else {
                is_left = false; is_right = false; insert_sd = 0;
            }
        } else {
            // v == cm->M (EL): use previous state, treat as left+right.
            let prv_v = tr.state[ti - 1];
            mode = tr.mode[ti - 1];
            sdl = 0; // prevents EL from affecting first/final_emit
            sdr = 0;
            is_left = true;
            is_right = true;
            nd = (1 + cm.ndidx[prv_v as usize]) as usize; // node EL replaced
            let ndt = cm.ndtype[nd] as i32;
            lpos = if ndt == MATP_nd || ndt == MATL_nd { emap.lpos[nd] } else { emap.lpos[nd] + 1 };
            rpos = if ndt == MATP_nd || ndt == MATR_nd { emap.rpos[nd] } else { emap.rpos[nd] - 1 };
            v = m; // (v already == m)
        }

        if is_left && (bounds_mode_emits_left(mode) || v == m) {
            cfrom_emit = cfrom_emit.min(lpos + insert_sd); // '+ insert_sd': insert after match/delete
            cto_emit = cto_emit.max(lpos);
            if sdl > 0 && cm.sttype[v as usize] as i32 != IL_ST {
                first_emit = first_emit.min(lpos);
                final_emit = final_emit.max(lpos);
            }
        }
        if is_right && (bounds_mode_emits_right(mode) || v == m) {
            cfrom_emit = cfrom_emit.min(rpos + insert_sd);
            cto_emit = cto_emit.max(rpos);
            if sdr > 0 && cm.sttype[v as usize] as i32 != IR_ST {
                first_emit = first_emit.min(rpos);
                final_emit = final_emit.max(rpos);
            }
        }
    }

    // Define cfrom_span/cto_span from the node the parsetree is rooted at
    // (tr->state[1]); guess for truncated hits (cm_parsetree.c:2705).
    let mut nd = cm.ndidx[tr.state[1] as usize] as usize;
    let ndt0 = cm.ndtype[nd] as i32;
    cfrom_span = if ndt0 == MATP_nd || ndt0 == MATL_nd { emap.lpos[nd] } else { emap.lpos[nd] + 1 };
    cto_span = if ndt0 == MATP_nd || ndt0 == MATR_nd { emap.rpos[nd] } else { emap.rpos[nd] - 1 };

    if tr.pass_idx == crate::cm_trunc::PLI_PASS_5P_ONLY_FORCE && have_i0 {
        let mut rpos = cto_span;
        while rpos == cto_span && nd > 0 {
            nd -= 1;
            let ndt = cm.ndtype[nd] as i32;
            rpos = if ndt == MATP_nd || ndt == MATR_nd { emap.rpos[nd] } else { emap.rpos[nd] - 1 };
        }
        let ndt = cm.ndtype[nd] as i32;
        cfrom_span = if ndt == MATP_nd || ndt == MATL_nd { emap.lpos[nd] } else { emap.lpos[nd] + 1 };
    }
    if tr.pass_idx == crate::cm_trunc::PLI_PASS_3P_ONLY_FORCE && have_j0 {
        let mut lpos = cfrom_span;
        while lpos == cfrom_span && nd > 0 {
            nd -= 1;
            let ndt = cm.ndtype[nd] as i32;
            lpos = if ndt == MATP_nd || ndt == MATL_nd { emap.lpos[nd] } else { emap.lpos[nd] + 1 };
        }
        let ndt = cm.ndtype[nd] as i32;
        cto_span = if ndt == MATP_nd || ndt == MATR_nd { emap.rpos[nd] } else { emap.rpos[nd] - 1 };
    }
    if tr.pass_idx == PLI_PASS_5P_AND_3P_FORCE && have_i0 && have_j0 {
        cfrom_span = 1;
        cto_span = cm.clen;
    }
    if tr.pass_idx == crate::cm_trunc::PLI_PASS_5P_AND_3P_ANY {
        cfrom_span = 1;
        cto_span = cm.clen;
    }
    // (C's PLI_PASS_STD_ANY branch is a pure sanity check: cfrom_emit==cfrom_span
    //  and cto_emit==cto_span; no value change.)

    CmBounds { cfrom_span, cto_span, cfrom_emit, cto_emit, first_emit, final_emit }
}
