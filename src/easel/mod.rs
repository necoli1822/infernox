//! easel - Rust implementation
//!
//! 1:1 port of the original C implementation.

// Faithful port: identifiers intentionally mirror the C source names (e.g. K, Kp).
#![allow(non_snake_case)]

pub mod alphabet;
pub mod constants;
pub mod distance;
pub mod error;
pub mod exponential;
pub mod gumbel;
pub mod histogram;
pub mod msa;
pub mod msafile;
pub mod random;
pub mod sqio;
pub mod ssi;
pub mod tree;

// Re-export commonly used types
pub use alphabet::EslAlphabet;
pub use error::{InfernalError, Result};
pub use exponential::{
    esl_exp_cdf, esl_exp_generic_cdf, esl_exp_generic_surv, esl_exp_logsurv, esl_exp_surv,
    esl_exp_FitComplete, esl_exp_FitCompleteScale,
};
pub use gumbel::{
    esl_gumbel_cdf, esl_gumbel_fit_complete, esl_gumbel_fit_complete_loc, esl_gumbel_invcdf,
    esl_gumbel_invsurv, esl_gumbel_logcdf, esl_gumbel_logpdf, esl_gumbel_logsurv, esl_gumbel_pdf,
    esl_gumbel_surv,
};
pub use histogram::{bin2lbound, bin2ubound, DatasetIs, EslHistogram};
pub use msa::EslMsa;
pub use msafile::{esl_msafile_write, read_all, MsaFormat};
pub use random::EslRandom;
pub use sqio::{esl_sqio_encode_format, guess_file_format, read_seqfile, SqFormat};
pub use ssi::{EslNewSsi, EslSsi, SsiEntry, ESL_SSI_FASTSUBSEQ};
