//! Infernal constants - 1:1 port from infernal.h
//!
//! State types, node types, score constants, and default parameters.

// =============================================================================
// State Types (D_st=0 ~ EL_st=9)
// =============================================================================

pub const D_ST: i32 = 0;
pub const MP_ST: i32 = 1;
pub const ML_ST: i32 = 2;
pub const MR_ST: i32 = 3;
pub const IL_ST: i32 = 4;
pub const IR_ST: i32 = 5;
pub const S_ST: i32 = 6;
pub const E_ST: i32 = 7;
pub const B_ST: i32 = 8;
pub const EL_ST: i32 = 9;

// =============================================================================
// Node Types - must match C Infernal's infernal.h exactly
// =============================================================================

pub const BIF_ND: i32 = 0;
pub const MATP_ND: i32 = 1;
pub const MATL_ND: i32 = 2;
pub const MATR_ND: i32 = 3;
pub const BEGL_ND: i32 = 4;
pub const BEGR_ND: i32 = 5;
pub const ROOT_ND: i32 = 6;
pub const END_ND: i32 = 7;

// Aliases for lowercase naming (used in some contexts)
pub const MATP_nd: i32 = MATP_ND;
pub const MATL_nd: i32 = MATL_ND;
pub const MATR_nd: i32 = MATR_ND;

// =============================================================================
// Unique State IDs (from infernal.h)
// These uniquely identify each state type within each node type.
// =============================================================================

pub const UNIQUESTATES: i32 = 21;
pub const NODETYPES: i32 = 8;

// ROOT node state IDs (0-2)
pub const ROOT_S: i32 = 0;
pub const ROOT_IL: i32 = 1;
pub const ROOT_IR: i32 = 2;

// BEGL node state IDs (3)
pub const BEGL_S: i32 = 3;

// BEGR node state IDs (4-5)
pub const BEGR_S: i32 = 4;
pub const BEGR_IL: i32 = 5;

// MATP node state IDs (6-11)
pub const MATP_MP: i32 = 6;
pub const MATP_ML: i32 = 7;
pub const MATP_MR: i32 = 8;
pub const MATP_D: i32 = 9;
pub const MATP_IL: i32 = 10;
pub const MATP_IR: i32 = 11;

// MATL node state IDs (12-14)
pub const MATL_ML: i32 = 12;
pub const MATL_D: i32 = 13;
pub const MATL_IL: i32 = 14;

// MATR node state IDs (15-17)
pub const MATR_MR: i32 = 15;
pub const MATR_D: i32 = 16;
pub const MATR_IR: i32 = 17;

// BIF node state ID (18)
pub const BIF_B: i32 = 18;

// END node state ID (19)
pub const END_E: i32 = 19;

// EL (End Local) state ID (20)
pub const EL: i32 = 20;

// =============================================================================
// Score Constants
// =============================================================================

pub const IMPOSSIBLE: f64 = -1e36;
pub const IMPOSSIBLE_F32: f32 = -1e7;  // f32 version (IMPOSSIBLE as f32 would overflow)
pub const MAXSCOREVAL: f64 = 1e35;
pub const IMPROBABLE: f64 = -5e35;
pub const INFTY: i32 = 987654321;
pub const INTSCALE: f64 = 1000.0;

// =============================================================================
// Connection Constants
// =============================================================================

pub const MAXCONNECT: i32 = 6;

// =============================================================================
// Default Parameters - Beta and Tau
// =============================================================================

pub const DEFAULT_BETA_W: f64 = 1e-7;
pub const DEFAULT_BETA_QDB1: f64 = 1e-7;
pub const DEFAULT_BETA_QDB2: f64 = 1e-15;
pub const DEFAULT_TAU: f64 = 1e-7;

// =============================================================================
// Default Parameters - Begin/End Probabilities
// =============================================================================

pub const DEFAULT_PBEGIN: f64 = 0.05;
pub const DEFAULT_PEND: f64 = 0.05;

// =============================================================================
// Default Parameters - E-value Targets
// =============================================================================

pub const DEFAULT_ETARGET: f64 = 0.59;
pub const DEFAULT_ETARGET_HMMFILTER: f64 = 0.38;

// =============================================================================
// Default Parameters - Null Model Probabilities
// =============================================================================

pub const DEFAULT_NULL2_OMEGA: f64 = 0.000015258791;
pub const DEFAULT_NULL3_OMEGA: f64 = 0.000015258791;

// =============================================================================
// Default Parameters - EL Self Probability
// =============================================================================

pub const DEFAULT_EL_SELFPROB: f64 = 0.94;

// =============================================================================
// Default Parameters - Max Tau
// =============================================================================

pub const DEFAULT_MAXTAU: f64 = 0.1;

// =============================================================================
// Default Parameters - CP9 Bands Thresholds
// =============================================================================

pub const DEFAULT_CP9BANDS_THRESH1: f64 = 0.01;
pub const DEFAULT_CP9BANDS_THRESH2: f64 = 0.98;

// =============================================================================
// Default Parameters - HMM Banded Matrix Size
// =============================================================================

pub const DEFAULT_HB_MXSIZE_MAX_MB: f64 = 1024.0;
pub const DEFAULT_HB_MXSIZE_MAX_W: f64 = 3000.0;
pub const DEFAULT_HB_MXSIZE_MIN_MB: f64 = 256.0;
pub const DEFAULT_HB_MXSIZE_MIN_W: f64 = 1000.0;

// =============================================================================
// Default Parameters - Ribosum Matrix
// =============================================================================

pub const DEFAULT_RMATRIX: &str = "RIBOSUM85-60";
pub const DEFAULT_RALPHA: f64 = 10.0;
pub const DEFAULT_RBETA: f64 = 5.0;
pub const DEFAULT_RALPHAP: f64 = 0.0;
pub const DEFAULT_RBETAP: f64 = 15.0;

// =============================================================================
// Default Parameters - Begin/End Scores
// =============================================================================

pub const DEFAULT_RBEGINSC: f64 = -0.01;
pub const DEFAULT_RENDSC: f64 = -15.0;

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() / (a.abs().max(b.abs()).max(1.0)) < EPSILON
    }

    #[test]
    fn test_state_types() {
        assert_eq!(D_ST, 0);
        assert_eq!(MP_ST, 1);
        assert_eq!(ML_ST, 2);
        assert_eq!(MR_ST, 3);
        assert_eq!(IL_ST, 4);
        assert_eq!(IR_ST, 5);
        assert_eq!(S_ST, 6);
        assert_eq!(E_ST, 7);
        assert_eq!(B_ST, 8);
        assert_eq!(EL_ST, 9);
    }

    #[test]
    fn test_node_types() {
        // Must match C Infernal's infernal.h
        assert_eq!(BIF_ND, 0);
        assert_eq!(MATP_ND, 1);
        assert_eq!(MATL_ND, 2);
        assert_eq!(MATR_ND, 3);
        assert_eq!(BEGL_ND, 4);
        assert_eq!(BEGR_ND, 5);
        assert_eq!(ROOT_ND, 6);
        assert_eq!(END_ND, 7);
    }

    #[test]
    fn test_score_constants() {
        assert!(approx_eq(IMPOSSIBLE, -1e36));
        assert!(approx_eq(MAXSCOREVAL, 1e35));
        assert!(approx_eq(IMPROBABLE, -5e35));
        assert_eq!(INFTY, 987654321);
        assert!(approx_eq(INTSCALE, 1000.0));
    }

    #[test]
    fn test_connection_constants() {
        assert_eq!(MAXCONNECT, 6);
        assert_eq!(UNIQUESTATES, 21);
    }

    #[test]
    fn test_default_beta_tau() {
        assert!(approx_eq(DEFAULT_BETA_W, 1e-7));
        assert!(approx_eq(DEFAULT_BETA_QDB1, 1e-7));
        assert!(approx_eq(DEFAULT_BETA_QDB2, 1e-15));
        assert!(approx_eq(DEFAULT_TAU, 1e-7));
    }

    #[test]
    fn test_default_begin_end() {
        assert!(approx_eq(DEFAULT_PBEGIN, 0.05));
        assert!(approx_eq(DEFAULT_PEND, 0.05));
    }

    #[test]
    fn test_default_el_selfprob() {
        assert!(approx_eq(DEFAULT_EL_SELFPROB, 0.94));
    }

    #[test]
    fn test_default_matrix_size() {
        assert!(approx_eq(DEFAULT_HB_MXSIZE_MAX_MB, 1024.0));
        assert!(approx_eq(DEFAULT_HB_MXSIZE_MAX_W, 3000.0));
        assert!(approx_eq(DEFAULT_HB_MXSIZE_MIN_MB, 256.0));
        assert!(approx_eq(DEFAULT_HB_MXSIZE_MIN_W, 1000.0));
    }

    #[test]
    fn test_ribosum_defaults() {
        assert_eq!(DEFAULT_RMATRIX, "RIBOSUM85-60");
        assert!(approx_eq(DEFAULT_RALPHA, 10.0));
        assert!(approx_eq(DEFAULT_RBETA, 5.0));
        assert!(approx_eq(DEFAULT_RALPHAP, 0.0));
        assert!(approx_eq(DEFAULT_RBETAP, 15.0));
    }
}
