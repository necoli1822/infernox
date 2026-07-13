//! Easel error types - 1:1 port from easel.h
//!
//! Error handling using thiserror for idiomatic Rust errors.

use thiserror::Error;

/// InfernalError - comprehensive error type for Easel/Infernal operations
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum InfernalError {
    #[error("generic failure")]
    Fail,

    #[error("end of line")]
    Eol,

    #[error("end of file")]
    Eof,

    #[error("end of data")]
    Eod,

    #[error("memory allocation failure")]
    Mem,

    #[error("not found")]
    NotFound,

    #[error("format error")]
    Format,

    #[error("ambiguous")]
    Ambiguous,

    #[error("division by zero")]
    DivZero,

    #[error("incompatible")]
    Incompat,

    #[error("invalid argument")]
    Inval,

    #[error("system error")]
    Sys,

    #[error("data corruption")]
    Corrupt,

    #[error("inconceivable error")]
    Inconceivable,

    #[error("syntax error")]
    Syntax,

    #[error("value out of range")]
    Range,

    #[error("duplicate")]
    Dup,

    #[error("no halt state")]
    NoHalt,

    #[error("no result")]
    NoResult,

    #[error("no data")]
    NoData,

    #[error("type mismatch")]
    Type,

    #[error("overwrite")]
    Overwrite,

    #[error("no space")]
    NoSpace,

    #[error("unimplemented")]
    Unimplemented,

    #[error("no format")]
    NoFormat,

    #[error("no alphabet")]
    NoAlphabet,

    #[error("write error")]
    Write,

    #[error("inaccurate result")]
    Inaccurate,
}

/// Result type alias for Infernal operations
pub type Result<T> = std::result::Result<T, InfernalError>;

impl InfernalError {
    /// Convert error code (i32) to InfernalError
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => None, // eslOK - no error
            1 => Some(InfernalError::Fail),
            2 => Some(InfernalError::Eol),
            3 => Some(InfernalError::Eof),
            4 => Some(InfernalError::Eod),
            5 => Some(InfernalError::Mem),
            6 => Some(InfernalError::NotFound),
            7 => Some(InfernalError::Format),
            8 => Some(InfernalError::Ambiguous),
            9 => Some(InfernalError::DivZero),
            10 => Some(InfernalError::Incompat),
            11 => Some(InfernalError::Inval),
            12 => Some(InfernalError::Sys),
            13 => Some(InfernalError::Corrupt),
            14 => Some(InfernalError::Inconceivable),
            15 => Some(InfernalError::Syntax),
            16 => Some(InfernalError::Range),
            17 => Some(InfernalError::Dup),
            18 => Some(InfernalError::NoHalt),
            19 => Some(InfernalError::NoResult),
            20 => Some(InfernalError::NoData),
            21 => Some(InfernalError::Type),
            22 => Some(InfernalError::Overwrite),
            23 => Some(InfernalError::NoSpace),
            24 => Some(InfernalError::Unimplemented),
            25 => Some(InfernalError::NoFormat),
            26 => Some(InfernalError::NoAlphabet),
            27 => Some(InfernalError::Write),
            28 => Some(InfernalError::Inaccurate),
            _ => Some(InfernalError::Inconceivable),
        }
    }

    /// Convert InfernalError to error code (i32)
    pub fn to_code(&self) -> i32 {
        match self {
            InfernalError::Fail => 1,
            InfernalError::Eol => 2,
            InfernalError::Eof => 3,
            InfernalError::Eod => 4,
            InfernalError::Mem => 5,
            InfernalError::NotFound => 6,
            InfernalError::Format => 7,
            InfernalError::Ambiguous => 8,
            InfernalError::DivZero => 9,
            InfernalError::Incompat => 10,
            InfernalError::Inval => 11,
            InfernalError::Sys => 12,
            InfernalError::Corrupt => 13,
            InfernalError::Inconceivable => 14,
            InfernalError::Syntax => 15,
            InfernalError::Range => 16,
            InfernalError::Dup => 17,
            InfernalError::NoHalt => 18,
            InfernalError::NoResult => 19,
            InfernalError::NoData => 20,
            InfernalError::Type => 21,
            InfernalError::Overwrite => 22,
            InfernalError::NoSpace => 23,
            InfernalError::Unimplemented => 24,
            InfernalError::NoFormat => 25,
            InfernalError::NoAlphabet => 26,
            InfernalError::Write => 27,
            InfernalError::Inaccurate => 28,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_roundtrip() {
        for code in 1..=28 {
            let err = InfernalError::from_code(code).unwrap();
            assert_eq!(err.to_code(), code);
        }
    }

    #[test]
    fn test_ok_returns_none() {
        assert!(InfernalError::from_code(0).is_none());
    }

    #[test]
    fn test_error_display() {
        assert_eq!(format!("{}", InfernalError::Mem), "memory allocation failure");
        assert_eq!(format!("{}", InfernalError::Eof), "end of file");
    }
}
