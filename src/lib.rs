//! infernal - Rust implementation
//!
//! 1:1 port of the original C implementation.

// Faithful C port: identifiers intentionally mirror the C source names (K/Kp,
// MATP_nd, ...), and the pre-parity `legacy/` modules retain unused/dead
// scaffolding kept only for compilation. These are intentional, not defects.
#![allow(dead_code, unused, non_snake_case, non_upper_case_globals)]

pub mod easel;
pub mod hmmer;

// ===== Verified faithful C-parity port (byte-for-byte checked) =====
pub mod cm_pipeline; // F1/F3/F3b filter cascade + f3_filter_sequence
pub mod p7_generic; // F4/F5 glocal envelope definition
pub mod cp9; // CP9 layer (fresh from-scratch; ilogsum done)
pub mod cm_search; // reusable faithful cmsearch library entry point (bin wraps this)
pub mod search_cli; // faithful esl_getopts-style CLI parser for cmsearch/cmscan bins
pub mod cm_nohmm; // faithful `-g --nohmm` global CM search (CYK filter + integer Inside)
pub mod cm_alidisplay; // faithful per-hit cm_alidisplay for the `-g --nohmm` path
pub mod cm_tophits; // tabular-output footer (cm_tophits_TabularTail) shared by cmsearch/cmscan
pub mod cm_trunc; // faithful truncated CYK (TrCYK) for the `-g` truncated passes
pub mod cm_dpsearch_trunc; // faithful QDB truncated scanners RefTrCYKScan / RefITrInsideScan
pub mod cm_dpalign; // faithful non-banded CM alignment DP (cmalign --nonbanded)
pub mod cm_dpsmall; // faithful divide-and-conquer exact CYK (cmalign --small)
pub mod truncyk; // faithful truncated divide-and-conquer CYK (TrCYK_DnC, cmbuild --refine --nonbanded)
pub mod cm_submodel; // faithful sub-CM alignment support (cmalign --sub)
pub mod cm_alndata; // reusable alignment config + DispatchSqAlignment (cmalign + cmbuild --refine)
pub mod cm_align_size; // DP matrix size estimation for cmalign score-report "mem (Mb)" column

// ===== Foundation shared by the faithful port =====
pub mod cm;
// cmbuild CM-construction pipeline (faithful port of cm_modelmaker.c / prior.c /
// eweight.c / display.c consensus / cm.c CMRebalance). Additive; used by bin/cmbuild.
pub mod cm_modelmaker;
pub mod cm_consensus;
pub mod cm_rebalance;
pub mod rsearch; // cmbuild --rsearch: RIBOSUM matrix reader + RSEARCH parameterization
pub mod prior;
pub mod eweight;
pub mod msaweight;
pub mod cm_emit; // cmemit sequence-emission core (EmitParsetree + FAST/LCG RNG)
pub mod cm_calibrate; // cmcalibrate: exp-tail E-value calibration (genomic HMM + MT RNG + fit)
pub mod cm_file;
pub mod constants;
pub mod evalue;
pub mod p7_hmm;
pub mod types;

// ===== cmbuild p7 filter-HMM build + calibration (task #70) =====
// Faithful port of HMMER3 p7_Builder for cmbuild's default filter-HMM path.
pub mod p7_builder; // Stage A: filter-HMM transitions + eff_nseq + MAXL
pub mod p7_filter_emit; // Stage B: filter-HMM emission construction (temp-CM mlp7 emissions)
pub mod p7_emit; // HMMER p7 sequence sampling (ProfileEmit/CoreEmit/FancyConsensus) for cmemit --hmmonly
pub mod p7_vitfilter; // faithful p7_ViterbiFilter (16-bit striped-equivalent) for calibration
pub mod p7_fwdback; // full-matrix odds-space Forward/Backward/Decoding for --hmmonly (local p7)
pub mod p7_omx; // striped-faithful Backward + posterior Decoding/DomainDecoding for --hmmonly
pub mod p7_domaindef; // LOCAL p7 domain definition (OA + Null2 + stochastic ensemble) for --hmmonly
pub mod p7_hmmonly; // --hmmonly final-stage logic (alidisplay + scoring + hits), self-contained
pub mod p7_oprofile; // SSE striped optimized-profile writer (.i1f/.i1p) for cmpress
pub mod cm_p7_calibrate; // Stage C: faithful cm_p7_Calibrate: p7 filter-HMM E-value calibration

// ===== Promoted from the former src/legacy/ (faithful-audited, C-parity verified) =====
// These three were the only pre-parity modules reachable from the faithful cmsearch
// path; each has been audited/verified against the C source (cm_qdband carries the
// BandTruncationNegligible C-parity fix, 16S-verified). The rest of legacy/ was dead
// (unreferenced by the faithful path or the live cmsearch bin) and has been deleted.
pub mod cm_qdband; // QDB band calculation (cm_qdband.c)
pub mod parsetree; // parse tree used by cm_alidisplay / cm_trunc (parsetree.c)
pub mod cm_emitmap; // consensus emit map used by cm_file (cm_emitmap.c)

// Re-export commonly used types (faithful modules + the 3 promoted)
pub use cm::CM;
pub use cm_file::{cm_file_read, cm_file_read_binary, cm_file_write_ascii, cm_file_write_binary};
pub use evalue::{ExpParams, P7ExpParams, calculate_bias_score, calculate_gc_content,
                 calculate_null3_bias, apply_null3_correction, score_to_evalue_corrected, NULL3_OMEGA,
                 log_sum2, get_composition, score_correction_null3, score_correction_null3_comp_unknown};
pub use p7_hmm::{P7EvParams, P7Profile, p7_hmmfile_read_binary, p7_hmmfile_write_ascii, p7_hmmfile_write_binary};
pub use cm_search::{FaithfulSearcher, FaithfulHit, FaithfulConfig};
pub use types::{EslDsq, EslPos};
pub use parsetree::Parsetree;
pub use cm_emitmap::{CMEmitMap, create_emit_map};
pub use cm_qdband::{
    QdbInfo, QdbSetBy, QdbResult,
    state_delta, state_left_delta, state_right_delta,
    cm_calculate_local_begin_probs, band_truncation_negligible,
    band_calculation_engine, calculate_query_dependent_bands, expand_bands
};
