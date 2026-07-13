//! Easel constants - 1:1 port from easel.h
//!
//! Error codes, mathematical constants, and digital sequence flags.

// =============================================================================
// Error Codes (eslOK=0 ~ eslEWRITE=27, eslEINACCURATE=28)
// =============================================================================

pub const ESL_OK: i32 = 0;
pub const ESL_FAIL: i32 = 1;
pub const ESL_EOL: i32 = 2;
pub const ESL_EOF: i32 = 3;
pub const ESL_EOD: i32 = 4;
pub const ESL_EMEM: i32 = 5;
pub const ESL_ENOTFOUND: i32 = 6;
pub const ESL_EFORMAT: i32 = 7;
pub const ESL_EAMBIGUOUS: i32 = 8;
pub const ESL_EDIVZERO: i32 = 9;
pub const ESL_EINCOMPAT: i32 = 10;
pub const ESL_EINVAL: i32 = 11;
pub const ESL_ESYS: i32 = 12;
pub const ESL_ECORRUPT: i32 = 13;
pub const ESL_EINCONCEIVABLE: i32 = 14;
pub const ESL_ESYNTAX: i32 = 15;
pub const ESL_ERANGE: i32 = 16;
pub const ESL_EDUP: i32 = 17;
pub const ESL_ENOHALT: i32 = 18;
pub const ESL_ENORESULT: i32 = 19;
pub const ESL_ENODATA: i32 = 20;
pub const ESL_ETYPE: i32 = 21;
pub const ESL_EOVERWRITE: i32 = 22;
pub const ESL_ENOSPACE: i32 = 23;
pub const ESL_EUNIMPLEMENTED: i32 = 24;
pub const ESL_ENOFORMAT: i32 = 25;
pub const ESL_ENOALPHABET: i32 = 26;
pub const ESL_EWRITE: i32 = 27;
pub const ESL_EINACCURATE: i32 = 28;

// =============================================================================
// Error Buffer Size
// =============================================================================

pub const ESL_ERRBUFSIZE: usize = 128;

// =============================================================================
// Boolean Constants
// =============================================================================

pub const TRUE: i32 = 1;
pub const FALSE: i32 = 0;

// =============================================================================
// Mathematical Constants
// =============================================================================

pub const ESL_CONST_E: f64 = 2.71828182845904523536028747135;
pub const ESL_CONST_PI: f64 = 3.14159265358979323846264338328;
pub const ESL_CONST_EULER: f64 = 0.57721566490153286060651209008;
pub const ESL_CONST_GOLD: f64 = 1.61803398874989484820458683437;
pub const ESL_CONST_LOG2: f64 = 0.69314718055994529;
pub const ESL_CONST_LOG2R: f64 = 1.44269504088896341;

// =============================================================================
// Numerical Approximation Thresholds
// =============================================================================

pub const ESL_SMALLX1: f64 = 5e-9;

// =============================================================================
// Digital Sequence (DSQ) Flags
// =============================================================================

pub const ESL_DSQ_SENTINEL: u8 = 255;
pub const ESL_DSQ_ILLEGAL: u8 = 254;
pub const ESL_DSQ_IGNORED: u8 = 253;
pub const ESL_DSQ_EOL: u8 = 252;
pub const ESL_DSQ_EOD: u8 = 251;

// =============================================================================
// File Path Constants
// =============================================================================

pub const ESL_DIRSLASH: char = '/';

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_values() {
        assert_eq!(ESL_OK, 0);
        assert_eq!(ESL_FAIL, 1);
        assert_eq!(ESL_EOF, 3);
        assert_eq!(ESL_EMEM, 5);
        assert_eq!(ESL_EWRITE, 27);
        assert_eq!(ESL_EINACCURATE, 28);
    }

    #[test]
    fn test_mathematical_constants() {
        assert!((ESL_CONST_E - std::f64::consts::E).abs() < 1e-15);
        assert!((ESL_CONST_PI - std::f64::consts::PI).abs() < 1e-15);
        assert!((ESL_CONST_LOG2 - std::f64::consts::LN_2).abs() < 1e-15);
    }

    #[test]
    fn test_dsq_flags() {
        assert_eq!(ESL_DSQ_SENTINEL, 255);
        assert_eq!(ESL_DSQ_ILLEGAL, 254);
        assert_eq!(ESL_DSQ_IGNORED, 253);
        assert_eq!(ESL_DSQ_EOL, 252);
        assert_eq!(ESL_DSQ_EOD, 251);
    }
}
