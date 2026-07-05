//! easel - Rust implementation
//!
//! 1:1 port of the original C implementation.

// Faithful port: identifiers intentionally mirror the C source names (e.g. K, Kp).
#![allow(non_snake_case)]

pub mod alphabet;
pub mod constants;
pub mod error;
pub mod exponential;
pub mod gumbel;
pub mod random;

// Re-export commonly used types
pub use alphabet::EslAlphabet;
pub use error::{InfernalError, Result};
pub use gumbel::{
    esl_gumbel_cdf, esl_gumbel_invcdf, esl_gumbel_invsurv, esl_gumbel_logcdf, esl_gumbel_logpdf,
    esl_gumbel_logsurv, esl_gumbel_pdf, esl_gumbel_surv,
};
pub use random::EslRandom;

#[cfg(test)]
mod tests {
    mod golden_phase1;
    mod golden_phase2;
    mod golden_phase3;
}
