//! Phase 1 Golden Tests - Constants
//!
//! Tests for Infernal and Easel constants.
//! Golden data extracted from infernal.h and easel.h.

use std::fs;
use std::mem::size_of;

use crate::constants::*;

fn load_golden(filename: &str) -> String {
    fs::read_to_string(format!("../tests/golden/phase1/{}", filename))
        .expect("Failed to load golden file")
}

fn parse_constant(content: &str, name: &str) -> Option<String> {
    for line in content.lines() {
        if line.starts_with(&format!("{}=", name)) {
            return Some(line.split('=').nth(1)?.to_string());
        }
    }
    None
}

fn parse_i32(content: &str, name: &str) -> Option<i32> {
    parse_constant(content, name)?.parse().ok()
}

fn parse_f64(content: &str, name: &str) -> Option<f64> {
    parse_constant(content, name)?.parse().ok()
}

fn parse_u8(content: &str, name: &str) -> Option<u8> {
    parse_constant(content, name)?.parse().ok()
}

fn parse_usize(content: &str, name: &str) -> Option<usize> {
    parse_constant(content, name)?.parse().ok()
}

const EPSILON: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() / (a.abs().max(b.abs()).max(1.0)) < EPSILON
}

// =====================================================================
// State Type Constants Tests (Infernal)
// =====================================================================

#[test]
fn test_state_types() {
    let golden = load_golden("infernal_constants.txt");

    let d_st = parse_i32(&golden, "D_st").expect("D_st not found");
    let mp_st = parse_i32(&golden, "MP_st").expect("MP_st not found");
    let ml_st = parse_i32(&golden, "ML_st").expect("ML_st not found");
    let mr_st = parse_i32(&golden, "MR_st").expect("MR_st not found");
    let il_st = parse_i32(&golden, "IL_st").expect("IL_st not found");
    let ir_st = parse_i32(&golden, "IR_st").expect("IR_st not found");
    let s_st = parse_i32(&golden, "S_st").expect("S_st not found");
    let e_st = parse_i32(&golden, "E_st").expect("E_st not found");
    let b_st = parse_i32(&golden, "B_st").expect("B_st not found");
    let el_st = parse_i32(&golden, "EL_st").expect("EL_st not found");

    // Verify golden values
    assert_eq!(d_st, 0);
    assert_eq!(mp_st, 1);
    assert_eq!(ml_st, 2);
    assert_eq!(mr_st, 3);
    assert_eq!(il_st, 4);
    assert_eq!(ir_st, 5);
    assert_eq!(s_st, 6);
    assert_eq!(e_st, 7);
    assert_eq!(b_st, 8);
    assert_eq!(el_st, 9);

    // Verify Rust constants match golden
    assert_eq!(infernal::constants::D_ST, d_st);
    assert_eq!(infernal::constants::MP_ST, mp_st);
    assert_eq!(infernal::constants::ML_ST, ml_st);
    assert_eq!(infernal::constants::MR_ST, mr_st);
    assert_eq!(infernal::constants::IL_ST, il_st);
    assert_eq!(infernal::constants::IR_ST, ir_st);
    assert_eq!(infernal::constants::S_ST, s_st);
    assert_eq!(infernal::constants::E_ST, e_st);
    assert_eq!(infernal::constants::B_ST, b_st);
    assert_eq!(infernal::constants::EL_ST, el_st);
}

// =====================================================================
// Node Type Constants Tests (Infernal)
// =====================================================================

#[test]
fn test_node_types() {
    let golden = load_golden("infernal_constants.txt");

    let bif_nd = parse_i32(&golden, "BIF_nd").expect("BIF_nd not found");
    let matp_nd = parse_i32(&golden, "MATP_nd").expect("MATP_nd not found");
    let matl_nd = parse_i32(&golden, "MATL_nd").expect("MATL_nd not found");
    let matr_nd = parse_i32(&golden, "MATR_nd").expect("MATR_nd not found");
    let begl_nd = parse_i32(&golden, "BEGL_nd").expect("BEGL_nd not found");
    let begr_nd = parse_i32(&golden, "BEGR_nd").expect("BEGR_nd not found");
    let root_nd = parse_i32(&golden, "ROOT_nd").expect("ROOT_nd not found");
    let end_nd = parse_i32(&golden, "END_nd").expect("END_nd not found");

    // Verify golden values
    assert_eq!(bif_nd, 0);
    assert_eq!(matp_nd, 1);
    assert_eq!(matl_nd, 2);
    assert_eq!(matr_nd, 3);
    assert_eq!(begl_nd, 4);
    assert_eq!(begr_nd, 5);
    assert_eq!(root_nd, 6);
    assert_eq!(end_nd, 7);

    // Verify Rust constants match golden
    assert_eq!(infernal::constants::BIF_ND, bif_nd);
    assert_eq!(infernal::constants::MATP_ND, matp_nd);
    assert_eq!(infernal::constants::MATL_ND, matl_nd);
    assert_eq!(infernal::constants::MATR_ND, matr_nd);
    assert_eq!(infernal::constants::BEGL_ND, begl_nd);
    assert_eq!(infernal::constants::BEGR_ND, begr_nd);
    assert_eq!(infernal::constants::ROOT_ND, root_nd);
    assert_eq!(infernal::constants::END_ND, end_nd);
}

// =====================================================================
// Score Constants Tests (Infernal)
// =====================================================================

#[test]
fn test_score_constants() {
    let golden = load_golden("infernal_constants.txt");

    let impossible = parse_f64(&golden, "IMPOSSIBLE").expect("IMPOSSIBLE not found");
    let maxscoreval = parse_f64(&golden, "MAXSCOREVAL").expect("MAXSCOREVAL not found");
    let improbable = parse_f64(&golden, "IMPROBABLE").expect("IMPROBABLE not found");
    let infty = parse_f64(&golden, "INFTY").expect("INFTY not found");
    let intscale = parse_f64(&golden, "INTSCALE").expect("INTSCALE not found");

    // Verify golden values
    assert!(approx_eq(impossible, -1e36));
    assert!(approx_eq(maxscoreval, 1e35));
    assert!(approx_eq(improbable, -5e35));
    assert!(approx_eq(infty, 987654321.0));
    assert!(approx_eq(intscale, 1000.0));

    // Verify Rust constants match golden
    assert!(approx_eq(infernal::constants::IMPOSSIBLE, impossible));
    assert!(approx_eq(infernal::constants::MAXSCOREVAL, maxscoreval));
    assert!(approx_eq(infernal::constants::IMPROBABLE, improbable));
    assert_eq!(infernal::constants::INFTY as f64, infty);
    assert!(approx_eq(infernal::constants::INTSCALE, intscale));
}

// =====================================================================
// Connection Constants Tests (Infernal)
// =====================================================================

#[test]
fn test_connection_constants() {
    let golden = load_golden("infernal_constants.txt");

    let maxconnect = parse_i32(&golden, "MAXCONNECT").expect("MAXCONNECT not found");
    let uniquestates = parse_i32(&golden, "UNIQUESTATES").expect("UNIQUESTATES not found");

    // Verify golden values
    assert_eq!(maxconnect, 6);
    assert_eq!(uniquestates, 21);

    // Verify Rust constants match golden
    assert_eq!(infernal::constants::MAXCONNECT, maxconnect);
    assert_eq!(infernal::constants::UNIQUESTATES, uniquestates);
}

// =====================================================================
// Default Parameter Tests (Infernal)
// =====================================================================

#[test]
fn test_default_beta_tau() {
    let golden = load_golden("infernal_constants.txt");

    let beta_w = parse_f64(&golden, "DEFAULT_BETA_W").expect("DEFAULT_BETA_W not found");
    let beta_qdb1 = parse_f64(&golden, "DEFAULT_BETA_QDB1").expect("DEFAULT_BETA_QDB1 not found");
    let beta_qdb2 = parse_f64(&golden, "DEFAULT_BETA_QDB2").expect("DEFAULT_BETA_QDB2 not found");
    let tau = parse_f64(&golden, "DEFAULT_TAU").expect("DEFAULT_TAU not found");

    // Verify golden values
    assert!(approx_eq(beta_w, 1e-7));
    assert!(approx_eq(beta_qdb1, 1e-7));
    assert!(approx_eq(beta_qdb2, 1e-15));
    assert!(approx_eq(tau, 1e-7));

    // Verify Rust constants match golden
    assert!(approx_eq(infernal::constants::DEFAULT_BETA_W, beta_w));
    assert!(approx_eq(infernal::constants::DEFAULT_BETA_QDB1, beta_qdb1));
    assert!(approx_eq(infernal::constants::DEFAULT_BETA_QDB2, beta_qdb2));
    assert!(approx_eq(infernal::constants::DEFAULT_TAU, tau));
}

#[test]
fn test_default_begin_end() {
    let golden = load_golden("infernal_constants.txt");

    let pbegin = parse_f64(&golden, "DEFAULT_PBEGIN").expect("DEFAULT_PBEGIN not found");
    let pend = parse_f64(&golden, "DEFAULT_PEND").expect("DEFAULT_PEND not found");

    // Verify golden values
    assert!(approx_eq(pbegin, 0.05));
    assert!(approx_eq(pend, 0.05));

    // Verify Rust constants match golden
    assert!(approx_eq(infernal::constants::DEFAULT_PBEGIN, pbegin));
    assert!(approx_eq(infernal::constants::DEFAULT_PEND, pend));
}

#[test]
fn test_default_el_selfprob() {
    let golden = load_golden("infernal_constants.txt");

    let el_selfprob = parse_f64(&golden, "DEFAULT_EL_SELFPROB").expect("DEFAULT_EL_SELFPROB not found");

    // Verify golden values
    assert!(approx_eq(el_selfprob, 0.94));

    // Verify Rust constants match golden
    assert!(approx_eq(infernal::constants::DEFAULT_EL_SELFPROB, el_selfprob));
}

#[test]
fn test_default_matrix_size() {
    let golden = load_golden("infernal_constants.txt");

    let max_mb = parse_f64(&golden, "DEFAULT_HB_MXSIZE_MAX_MB").expect("DEFAULT_HB_MXSIZE_MAX_MB not found");
    let max_w = parse_f64(&golden, "DEFAULT_HB_MXSIZE_MAX_W").expect("DEFAULT_HB_MXSIZE_MAX_W not found");
    let min_mb = parse_f64(&golden, "DEFAULT_HB_MXSIZE_MIN_MB").expect("DEFAULT_HB_MXSIZE_MIN_MB not found");
    let min_w = parse_f64(&golden, "DEFAULT_HB_MXSIZE_MIN_W").expect("DEFAULT_HB_MXSIZE_MIN_W not found");

    // Verify golden values
    assert!(approx_eq(max_mb, 1024.0));
    assert!(approx_eq(max_w, 3000.0));
    assert!(approx_eq(min_mb, 256.0));
    assert!(approx_eq(min_w, 1000.0));

    // Verify Rust constants match golden
    assert!(approx_eq(infernal::constants::DEFAULT_HB_MXSIZE_MAX_MB, max_mb));
    assert!(approx_eq(infernal::constants::DEFAULT_HB_MXSIZE_MAX_W, max_w));
    assert!(approx_eq(infernal::constants::DEFAULT_HB_MXSIZE_MIN_MB, min_mb));
    assert!(approx_eq(infernal::constants::DEFAULT_HB_MXSIZE_MIN_W, min_w));
}

// =====================================================================
// Easel Error Codes Tests
// =====================================================================

#[test]
fn test_easel_error_codes() {
    let golden = load_golden("easel_constants.txt");

    // Parse all error codes from golden
    let esl_ok = parse_i32(&golden, "eslOK").expect("eslOK not found");
    let esl_fail = parse_i32(&golden, "eslFAIL").expect("eslFAIL not found");
    let esl_eol = parse_i32(&golden, "eslEOL").expect("eslEOL not found");
    let esl_eof = parse_i32(&golden, "eslEOF").expect("eslEOF not found");
    let esl_eod = parse_i32(&golden, "eslEOD").expect("eslEOD not found");
    let esl_emem = parse_i32(&golden, "eslEMEM").expect("eslEMEM not found");
    let esl_enotfound = parse_i32(&golden, "eslENOTFOUND").expect("eslENOTFOUND not found");
    let esl_eformat = parse_i32(&golden, "eslEFORMAT").expect("eslEFORMAT not found");
    let esl_eambiguous = parse_i32(&golden, "eslEAMBIGUOUS").expect("eslEAMBIGUOUS not found");
    let esl_edivzero = parse_i32(&golden, "eslEDIVZERO").expect("eslEDIVZERO not found");
    let esl_eincompat = parse_i32(&golden, "eslEINCOMPAT").expect("eslEINCOMPAT not found");
    let esl_einval = parse_i32(&golden, "eslEINVAL").expect("eslEINVAL not found");
    let esl_esys = parse_i32(&golden, "eslESYS").expect("eslESYS not found");
    let esl_ecorrupt = parse_i32(&golden, "eslECORRUPT").expect("eslECORRUPT not found");
    let esl_einconceivable = parse_i32(&golden, "eslEINCONCEIVABLE").expect("eslEINCONCEIVABLE not found");
    let esl_esyntax = parse_i32(&golden, "eslESYNTAX").expect("eslESYNTAX not found");
    let esl_erange = parse_i32(&golden, "eslERANGE").expect("eslERANGE not found");
    let esl_edup = parse_i32(&golden, "eslEDUP").expect("eslEDUP not found");
    let esl_enohalt = parse_i32(&golden, "eslENOHALT").expect("eslENOHALT not found");
    let esl_enoresult = parse_i32(&golden, "eslENORESULT").expect("eslENORESULT not found");
    let esl_enodata = parse_i32(&golden, "eslENODATA").expect("eslENODATA not found");
    let esl_etype = parse_i32(&golden, "eslETYPE").expect("eslETYPE not found");
    let esl_eoverwrite = parse_i32(&golden, "eslEOVERWRITE").expect("eslEOVERWRITE not found");
    let esl_enospace = parse_i32(&golden, "eslENOSPACE").expect("eslENOSPACE not found");
    let esl_eunimplemented = parse_i32(&golden, "eslEUNIMPLEMENTED").expect("eslEUNIMPLEMENTED not found");
    let esl_enoformat = parse_i32(&golden, "eslENOFORMAT").expect("eslENOFORMAT not found");
    let esl_enoalphabet = parse_i32(&golden, "eslENOALPHABET").expect("eslENOALPHABET not found");
    let esl_ewrite = parse_i32(&golden, "eslEWRITE").expect("eslEWRITE not found");
    let esl_einaccurate = parse_i32(&golden, "eslEINACCURATE").expect("eslEINACCURATE not found");

    // Verify Rust constants match golden
    assert_eq!(ESL_OK, esl_ok);
    assert_eq!(ESL_FAIL, esl_fail);
    assert_eq!(ESL_EOL, esl_eol);
    assert_eq!(ESL_EOF, esl_eof);
    assert_eq!(ESL_EOD, esl_eod);
    assert_eq!(ESL_EMEM, esl_emem);
    assert_eq!(ESL_ENOTFOUND, esl_enotfound);
    assert_eq!(ESL_EFORMAT, esl_eformat);
    assert_eq!(ESL_EAMBIGUOUS, esl_eambiguous);
    assert_eq!(ESL_EDIVZERO, esl_edivzero);
    assert_eq!(ESL_EINCOMPAT, esl_eincompat);
    assert_eq!(ESL_EINVAL, esl_einval);
    assert_eq!(ESL_ESYS, esl_esys);
    assert_eq!(ESL_ECORRUPT, esl_ecorrupt);
    assert_eq!(ESL_EINCONCEIVABLE, esl_einconceivable);
    assert_eq!(ESL_ESYNTAX, esl_esyntax);
    assert_eq!(ESL_ERANGE, esl_erange);
    assert_eq!(ESL_EDUP, esl_edup);
    assert_eq!(ESL_ENOHALT, esl_enohalt);
    assert_eq!(ESL_ENORESULT, esl_enoresult);
    assert_eq!(ESL_ENODATA, esl_enodata);
    assert_eq!(ESL_ETYPE, esl_etype);
    assert_eq!(ESL_EOVERWRITE, esl_eoverwrite);
    assert_eq!(ESL_ENOSPACE, esl_enospace);
    assert_eq!(ESL_EUNIMPLEMENTED, esl_eunimplemented);
    assert_eq!(ESL_ENOFORMAT, esl_enoformat);
    assert_eq!(ESL_ENOALPHABET, esl_enoalphabet);
    assert_eq!(ESL_EWRITE, esl_ewrite);
    assert_eq!(ESL_EINACCURATE, esl_einaccurate);
}

// =====================================================================
// Easel Mathematical Constants Tests
// =====================================================================

#[test]
fn test_easel_math_constants() {
    let golden = load_golden("easel_constants.txt");

    let const_e = parse_f64(&golden, "eslCONST_E").expect("eslCONST_E not found");
    let const_pi = parse_f64(&golden, "eslCONST_PI").expect("eslCONST_PI not found");
    let const_euler = parse_f64(&golden, "eslCONST_EULER").expect("eslCONST_EULER not found");
    let const_gold = parse_f64(&golden, "eslCONST_GOLD").expect("eslCONST_GOLD not found");
    let const_log2 = parse_f64(&golden, "eslCONST_LOG2").expect("eslCONST_LOG2 not found");
    let const_log2r = parse_f64(&golden, "eslCONST_LOG2R").expect("eslCONST_LOG2R not found");

    // Verify Rust constants match golden
    assert!(approx_eq(ESL_CONST_E, const_e));
    assert!(approx_eq(ESL_CONST_PI, const_pi));
    assert!(approx_eq(ESL_CONST_EULER, const_euler));
    assert!(approx_eq(ESL_CONST_GOLD, const_gold));
    assert!(approx_eq(ESL_CONST_LOG2, const_log2));
    assert!(approx_eq(ESL_CONST_LOG2R, const_log2r));
}

// =====================================================================
// Easel DSQ Flags Tests
// =====================================================================

#[test]
fn test_easel_dsq_flags() {
    let golden = load_golden("easel_constants.txt");

    let dsq_sentinel = parse_u8(&golden, "eslDSQ_SENTINEL").expect("eslDSQ_SENTINEL not found");
    let dsq_illegal = parse_u8(&golden, "eslDSQ_ILLEGAL").expect("eslDSQ_ILLEGAL not found");
    let dsq_ignored = parse_u8(&golden, "eslDSQ_IGNORED").expect("eslDSQ_IGNORED not found");
    let dsq_eol = parse_u8(&golden, "eslDSQ_EOL").expect("eslDSQ_EOL not found");
    let dsq_eod = parse_u8(&golden, "eslDSQ_EOD").expect("eslDSQ_EOD not found");

    // Verify Rust constants match golden
    assert_eq!(ESL_DSQ_SENTINEL, dsq_sentinel);
    assert_eq!(ESL_DSQ_ILLEGAL, dsq_illegal);
    assert_eq!(ESL_DSQ_IGNORED, dsq_ignored);
    assert_eq!(ESL_DSQ_EOL, dsq_eol);
    assert_eq!(ESL_DSQ_EOD, dsq_eod);
}

// =====================================================================
// Type Size Tests
// =====================================================================

#[test]
fn test_type_sizes() {
    let golden = load_golden("type_sizes.txt");

    // Parse expected sizes from golden
    let esl_dsq_size = parse_usize(&golden, "sizeof(ESL_DSQ)").expect("sizeof(ESL_DSQ) not found");
    let esl_pos_size = parse_usize(&golden, "sizeof(esl_pos_t)").expect("sizeof(esl_pos_t) not found");

    // Verify Rust types have correct sizes
    assert_eq!(size_of::<infernal::types::EslDsq>(), esl_dsq_size);
    assert_eq!(size_of::<infernal::types::EslPos>(), esl_pos_size);
}

// =====================================================================
// Error Buffer Size Test
// =====================================================================

#[test]
fn test_error_buffer_size() {
    let golden = load_golden("easel_constants.txt");

    let errbufsize = parse_usize(&golden, "eslERRBUFSIZE").expect("eslERRBUFSIZE not found");

    assert_eq!(ESL_ERRBUFSIZE, errbufsize);
}
