//! infernal - Rust implementation
//!
//! 1:1 port of the original C implementation.

// Faithful C port: identifiers intentionally mirror the C source names (K/Kp,
// MATP_nd, ...), and the pre-parity `legacy/` modules retain unused/dead
// scaffolding kept only for compilation. These are intentional, not defects.
#![allow(dead_code, unused, non_snake_case, non_upper_case_globals)]

// ===== Verified faithful C-parity port (byte-for-byte checked) =====
pub mod cm_pipeline; // F1/F3/F3b filter cascade + f3_filter_sequence
pub mod p7_generic; // F4/F5 glocal envelope definition
pub mod cp9_faithful; // CP9 layer (fresh from-scratch; ilogsum done)
pub mod faithful_search; // reusable faithful cmsearch library entry point (bin wraps this)
pub mod cm_nohmm; // faithful `-g --nohmm` global CM search (CYK filter + integer Inside)
pub mod cm_alidisplay; // faithful per-hit cm_alidisplay for the `-g --nohmm` path
pub mod cm_trunc; // faithful truncated CYK (TrCYK) for the `-g` truncated passes

// ===== Foundation shared by the faithful port =====
pub mod cm;
pub mod cm_file;
pub mod constants;
pub mod evalue;
pub mod p7_hmm;
pub mod types;

// ===== Legacy pre-parity code (src/legacy/), NOT verified faithful =====
// Re-exported flat for backward compatibility so existing `crate::<mod>` /
// `infernal::<mod>` paths keep resolving; prefer `crate::legacy::<mod>` in new code.
pub mod legacy;
pub use legacy::*;

// Re-export commonly used types
pub use alignment::{Alignment, AlignMode};
pub use cm::CM;
pub use cm_dp::{
    cyk_divide_and_conquer, cyk_inside, cyk_inside_score, cyk_inside_debug,
    cm_inside_with_matrix, cm_inside_scan, cm_outside, cm_posterior, cm_outside_align_hb, cm_posterior_hb,
    cm_cyk_scan, ScanHit
};
pub use pipeline::{cm_cyk_inside_align_hb, cm_inside_align_hb};
pub use cm_file::cm_file_read;
pub use cmsearch::{cmsearch, format_tblout, CmsearchResult};
pub use cp9::CP9;
pub use cp9_bands::{CP9Bands, QDBBands};
pub use cp9_dp::{cp9_forward, cp9_backward, cp9_posterior_decode, cp9_hmm_band_bounds, cp9_compute_bands, CP9Posterior};
pub use cp9_mx::{CP9MX, CP9Shadow, CP9Trace};
pub use evalue::{ExpParams, P7ExpParams, calculate_bias_score, calculate_gc_content,
                 calculate_null3_bias, apply_null3_correction, score_to_evalue_corrected, NULL3_OMEGA,
                 log_sum2, get_composition, score_correction_null3, score_correction_null3_comp_unknown};
pub use output::{StockholmOutput, TbloutLine, tblout_header};
pub use p7_hmm::{P7EvParams, P7Profile};
pub use p7_dp::{p7_msv_filter, p7_viterbi_filter, p7_forward_filter, nats_to_bits};
pub use p7_simd::{P7ProfileOpt, p7_msv_simd, p7_viterbi_simd, p7_forward_simd};
pub use parsetree::Parsetree;
pub use pipeline::{SearchHit, PipelineConfig, cm_search};
pub use faithful_search::{FaithfulSearcher, FaithfulHit, FaithfulConfig};
pub use types::{EslDsq, EslPos};
pub use cm_emitmap::{CMEmitMap, create_emit_map};
pub use cp9_map::{CP9Map, HmmStateType, cp9_map_cm2hmm};
pub use cm_subinfo::{TransitionMap, create_transition_map, cm_expected_state_occupancy, total_states_in_node, state_is_detached};
pub use cp9_modelmaker::{cplan9_from_cm, cplan9_set_local, cplan9_set_global};
pub use cm_qdband::{
    QdbInfo, QdbSetBy, QdbResult,
    state_delta, state_left_delta, state_right_delta,
    cm_calculate_local_begin_probs, band_truncation_negligible,
    band_calculation_engine, calculate_query_dependent_bands, expand_bands
};
pub use cm_dp_hb::{
    CMHbMx,
    cm_cyk_inside_align_hb_bands,
    cm_inside_align_hb_bands,
    cm_outside_align_hb_bands,
    cm_posterior_hb_bands
};
pub use simd::{simd_max_f32, simd_add_f32, simd_horizontal_max, simd_logsumexp, simd_msv_inner};
pub use truncated::{TruncMode, TruncatedHit, detect_truncation, apply_trunc_penalty};
